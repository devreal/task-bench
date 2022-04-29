#define TTG_USE_PARSEC 1

#include <cstring>
#include <memory>
#include <array>
#include <ttg.h>
#include <ttg/serialization/splitmd_data_descriptor.h>

#include "../core/core.h"
#include "../core/timer.h"

#define MAX_DEPS 10

struct Key {
  long x; // point in the grid
  long graph_id; // the graph to be used
  int timestep;

  Key() = default;

  Key(long x, long graph, int ts)
  : x(x), graph_id(graph), timestep(ts)
  { }

  Key next(int ts_delta = 1) const {
    return Key(x, graph_id, timestep+ts_delta);
  }

  bool operator==(const Key& k) {
    return k.x == x && k.graph_id == graph_id && k.timestep == timestep;
  }

  bool operator!=(const Key& k) {
    return !(*this == k);
  }

  size_t hash() {
    return (graph_id << 58) ^ (x << 24) ^ timestep; // there can only be one task per point per timestep
  }
};

namespace std {
  std::ostream& operator<<(std::ostream& s, const Key& k) {
    s << "Key(" << k.x << "," << k.graph_id << "," << k.timestep << ")";
    return s;
  }
} // namespace std



struct data_t {
private:

  struct deleter {
    bool m_allocated = false;
    deleter() = default;
    deleter(bool allocated) : m_allocated(allocated) { }
    void operator()(char ptr[]) {
      if (m_allocated) delete[] ptr;
    }
  };

  Key m_source_key; // the task from which this data originated
  size_t m_size;
  std::unique_ptr<char[], deleter> m_data = std::unique_ptr<char[], deleter>(nullptr, false);
public:

  struct metadata {
    Key key;
    size_t size;
  };

  data_t() = default;

  explicit data_t(const data_t& d)
  : m_source_key(d.m_source_key)
  , m_size(d.m_size)
  , m_data(new char[d.m_size], deleter(true))
  {
    std::memcpy(m_data.get(), d.m_data.get(), d.m_size);
  }

  data_t(data_t&& d) = default;

  data_t& operator=(const data_t& d) {
    if (this == &d) return *this;
    m_source_key = d.m_source_key;
    m_data = std::unique_ptr<char[], deleter>(new char[d.m_size], deleter(true));
    m_size = d.m_size;
    std::memcpy(m_data.get(), d.m_data.get(), d.m_size);
    return *this;
  }
  data_t& operator=(data_t&& d) = default;

  data_t(Key source, size_t size)
  : m_source_key(source)
  , m_size(size)
  , m_data(new char[size], deleter(true))
  { }

  data_t(Key source, char ptr[], size_t size)
  : m_source_key(source)
  , m_size(size)
  , m_data(ptr, deleter(false))
  { }

  explicit data_t(const metadata& meta)
  : m_source_key(meta.key)
  , m_size(meta.size)
  , m_data(new char[meta.size], deleter(true))
  { }

  const Key& source_key() const {
    return m_source_key;
  }

  size_t size() const {
    return m_size;
  }

  char* data() {
    return m_data.get();
  }

  const char* data() const {
    return m_data.get();
  }

  auto get_metadata() const {
    return metadata{m_source_key, m_size};
  }
};

struct local_graph_data {
private:
  std::vector<data_t> m_points;
  std::unique_ptr<char[]> m_ptr;
  long m_lb;
  long m_ub;
  long m_num_points;
  int m_nb_fields; // number of copies in time dimension (i.e., double buffering)
public:

  local_graph_data(long lb, long ub, int nb_fields, size_t point_size, int graph_index)
  : m_lb(lb)
  , m_ub(ub)
  , m_num_points(ub-lb)
  , m_nb_fields(nb_fields)
  {
    long p = 0;
    auto adj_point_size = std::max(64UL, point_size); // at least one cache line per task
    m_ptr = std::unique_ptr<char[]>(new char[adj_point_size*(ub-lb)*nb_fields]);
    for (int f = 0; f < nb_fields; ++f) {
      for (long x = lb; x < ub; ++x) {
        m_points.emplace_back(data_t(Key{x, graph_index, -1}, &m_ptr[p*adj_point_size], point_size));
        ++p;
        //std::cout << "point " << x << " data " << (void*) (--(local_data[graph.graph_index].points.end()))->data() << std::endl;
      }
    }
  }

  data_t& operator()(long x, int ts) {
    return m_points[(ts%m_nb_fields)*m_num_points + (x-m_lb)];
  }

  data_t&& acquire(long x, int ts) {
    return std::move(m_points[(ts%m_nb_fields)*m_num_points + (x-m_lb)]);
  }

  long num_points() const {
    return m_num_points;
  }

  long lb() const {
    return m_lb;
  }

  long ub() const {
    return m_ub;
  }

};

// support zero copy transfers
namespace ttg {

  template<>
  struct SplitMetadataDescriptor<data_t>
  {

    auto get_metadata(const data_t& t)
    {
      return t.get_metadata();
    }

    auto get_data(data_t& t)
    {
      return std::array<iovec, 1>({{{t.size(), t.data()}}});
    }

    auto create_from_metadata(const typename data_t::metadata& meta)
    {
      return data_t(meta);
    }
  };
}

struct TTGApp : App {
  TTGApp(int argc, char **argv);
  ~TTGApp();

  void execute_main_loop();

private:
  std::vector<local_graph_data> local_data; // data for each graph width
  ttg::World world;
  size_t max_scratch_bytes_per_task = 0;
  std::mutex tsmtx;
  std::vector<char*> tsdata;
  char **extra_local_memory;
};

TTGApp::TTGApp(int argc, char **argv)
: App(argc, argv)
{
  ttg::initialize(argc, argv);

  world = ttg::get_default_world();

  ttg::World world = ttg::default_execution_context();

  local_data.reserve(graphs.size());

  for (auto& graph : graphs) {
    long width = graph.max_width;
    int nprocs = world.size();
    int rank = world.rank();
    long lb = ((width+nprocs-1)/nprocs)*rank;
    // the generic number of points (might be different on the last process)
    long max_local_width = (width+nprocs-1)/nprocs;
    // the specific upper bound for this process
    long ub = std::min((max_local_width)*(rank+1), width);
    local_data.emplace_back(lb, ub, graph.nb_fields, graph.output_bytes_per_task, graph.graph_index);
#if 0
    long local_width = (ub - lb);
    local_data[graph.graph_index].lb = lb;
    local_data[graph.graph_index].ub = ub;
    local_data[graph.graph_index].points.reserve(ub-lb);
    local_data[graph.graph_index].num_points = local_width;
    auto num_points = graph.nb_fields * local_width;
    char *ptr = new char[(ub-lb)*point_size];
    local_data[graph.graph_index].ptr = ptr;
    for (long x = lb; x < ub; ++x) {
      local_data[graph.graph_index].points.emplace_back(data_t(Key{x, graph.graph_index, -1}, &ptr[(x-lb)*size], size));
      //std::cout << "point " << x << " data " << (void*) (--(local_data[graph.graph_index].points.end()))->data() << std::endl;
    }
#endif
    max_scratch_bytes_per_task = std::max(max_scratch_bytes_per_task, graph.scratch_bytes_per_task);
  }

  int cores = ttg::detail::num_threads();
  extra_local_memory = (char**)malloc(sizeof(char*) * cores);
  assert(extra_local_memory != NULL);
  for (int i = 0; i < cores; i++) {
    if (max_scratch_bytes_per_task > 0) {
      extra_local_memory[i] = (char*)malloc(sizeof(char)*max_scratch_bytes_per_task);
      TaskGraph::prepare_scratch(extra_local_memory[i], sizeof(char)*max_scratch_bytes_per_task);
    } else {
      extra_local_memory[i] = NULL;
    }
  }

}


TTGApp::~TTGApp()
{

  /* free scratch memory */
  for (auto& ptr : tsdata) {
    delete[] ptr;
  }
  tsdata.clear();

  ttg::finalize();

}

void TTGApp::execute_main_loop()
{

  if (world.rank() == 0) {
    display();
  }

  //sleep(10);

  /**
   * Tasks can send to their next incarnation in the next timestep (producer)
   * and to their neighbors (consumers). Input from neighbors is variable
   * (it can change between incarnations) and thus is aggregated.
   */
  ttg::Edge<Key, data_t> I2P; // initiator to producer
  ttg::Edge<Key, data_t> I2C; // initiator to consumer
  ttg::Edge<Key, data_t> P2P; // producer to producer
  ttg::Edge<Key, data_t> P2C; // producer to consumer
  ttg::Edge<Key, data_t> P2W; // producer to writeback

  /* 1D cyclic over the points */
  auto procmap = [&](const Key& key){
    //std::cout << "procmap: " << key.x << " -> " << key.x / local_data[key.graph_id].num_points << std::endl;
    return key.x / local_data[key.graph_id].num_points();
  };

  /* wrapper for iterating over a set of (reverse) dependencies
   * iteration stops if \c fn returns \c true. In that case,
   * the call returns true, false otherwise.
   */
  auto for_each_dep = [](TaskGraph& graph, long point, int timestep, bool rdeps, auto&& fn){

    auto dset = graph.dependence_set_at_timestep(timestep);
    /* std::pair is default initialized even on the stack, best to avoid */
    static thread_local std::pair<long, long> deps[MAX_DEPS];
    long offset = graph.offset_at_timestep(timestep);
    long width = graph.width_at_timestep(timestep);

    auto num_successors = (rdeps) ? graph.reverse_dependencies(dset, point, deps)
                                  : graph.dependencies(dset, point, deps);
    for (size_t k = 0; k < num_successors; ++k) {
      for (long x = std::max(deps[k].first, offset); x <= std::min(deps[k].second, width); x++) {
        if (fn(x)) return true;
      }
    }

    return false;
  };

  auto broadcast_keys = [&](const Key& key, data_t&& data, auto& out){
    TaskGraph& graph = graphs[key.graph_id];
    auto next_timestep = key.timestep+1;
    //std::cout << "broadcast: graph type " << graph.dependence << " timestep " << next_timestep << " of " << graph.timesteps << " point " << key.x << std::endl;
    if (next_timestep < graph.timesteps) {
      //std::vector<Key> consumers;
      Key consumers[MAX_DEPS];
      int c = 0;
      Key next_k = key.next(graph.nb_fields); // this task in next timestep
      bool have_next_k = false;
      if (graph.nb_fields == 1) {
        /* send to all consumers, except for same point in next timestep */
        for_each_dep(graph, key.x, next_timestep, true,
                    [&](long x) mutable {
                      if (x != key.x) {
                        /* send to other points consuming our results */
                        consumers[c++] = Key(x, key.graph_id, next_timestep);
                      } else {
                        have_next_k = true;
                      }
                      return false; // always continue iterating
                    });
      } else {
        /* send to all consumers, including same point in next timestep */
        for_each_dep(graph, key.x, next_timestep, true,
                    [&](long x) mutable {
                      consumers[c++] = Key(x, key.graph_id, next_timestep);
                      return false; // always continue iterating
                    });
      }

      if (!have_next_k) {
        /* we haven't found the next point at x so keep iterating over subsequent timesteps until we find it */
        for (int t = key.timestep+graph.nb_fields; t < graph.timesteps; t += graph.nb_fields) {
          long offset = graph.offset_at_timestep(t);
          long width = graph.width_at_timestep(t);
          //std::cout << "INIT: timestep " << t << " offset " << offset << " width " << width << " lb " << graph_data.lb << " ub " << graph_data.ub << std::endl;
          if (key.x >= offset && key.x < width) {
            next_k = Key(key.x, graph.graph_index, t);
            have_next_k = true;
            break;
          }
        }
      }

#if 0
      std::cout << "Broadcast from key " << key << " to";
      for (auto& k : consumers) {
        std::cout << " " << k;
      }
      if (have_next_k) {
        std::cout << ", next key " << next_k;
      }
      std::cout << std::endl;
#endif // 0

      assert(data.data() != nullptr);
      if (have_next_k) {
        /* broadcast copy to consumers and next same task */
        ttg::broadcast<1, 0>(std::make_tuple(ttg::span(consumers, consumers+c), next_k), std::move(data), out);
      } else if (c > 0) {
        /* broadcast to consumers only */
        ttg::broadcast<1>(ttg::span(consumers, consumers+c), std::move(data), out);
      }
    } else {
      ttg::send<2>(key, std::move(data), out);
    }
  };


  auto init_tt = ttg::make_tt<Key>(
    [&](const Key&, std::tuple<ttg::Out<Key, data_t>, ttg::Out<Key, data_t>>& out){
      for (auto& graph : graphs) {
        /* kick off init tasks for all points I own */
        auto& graph_data = local_data[graph.graph_index];
        size_t num_seen_x = 0;
        long lb = graph_data.lb();
        long ub = graph_data.ub();
        assert((ub-lb) == graph_data.num_points());
        long num_x = graph_data.num_points()*graph.nb_fields;
        std::vector<bool> seen_x(num_x);
        for (int t = 0; t < graph.timesteps; ++t) {
          long offset = graph.offset_at_timestep(t);
          long width = graph.width_at_timestep(t);
          //std::cout << "INIT: timestep " << t << " offset " << offset << " width " << width << " lb " << graph_data.lb << " ub " << graph_data.ub << std::endl;
          for (long x = std::max(lb, offset); x < std::min(ub, width); x++) {
            auto seen_idx = (t % graph.nb_fields)*graph_data.num_points() + (x-lb);
            if (!seen_x[seen_idx]) {
              Key key = Key(x, graph.graph_index, t);
              //std::cout << "SEND to key " << key << std::endl;
              ttg::send<0>(key, graph_data.acquire(x, t), out);
              seen_x[seen_idx] = true;
              ++num_seen_x;
            }
            if (num_x == num_seen_x) return;
          }
        }
      }
    },
    ttg::edges(), ttg::edges(I2P, I2C), "Init", {}, {"I2P", "I2C"});

  /* init should execute on all processes */
  init_tt->set_keymap([&](const Key&){ return world.rank(); });

  auto point_tt = ttg::make_tt(
    /* data is our modifyable data, cdata is from other inputs but ourselves */
    [&](const Key& key,
        data_t&& data,
        const ttg::Aggregator<data_t>& cdata,
        std::tuple<ttg::Out<Key, data_t>, ttg::Out<Key, data_t>,  ttg::Out<Key, data_t>>& out) {

      TaskGraph& graph = graphs[key.graph_id];

      //std::cout << "Point " << key << std::endl;

      char *output_ptr = data.data();
      size_t output_bytes = graph.output_bytes_per_task;
      const char* input_ptrs[MAX_DEPS];
      size_t input_bytes[MAX_DEPS];
      /* populate the input to the kernel */
      {
        auto insert_data = [&](auto& data){
          long pos = 0;
          for_each_dep(graph, key.x, key.timestep, false,
            [&](long x){
              if (x == data.source_key().x) {
                input_ptrs[pos] = data.data();
                input_bytes[pos] = output_bytes;
                return true; // we're done
              }
              pos++;
              return false; // keep iterating
            }
          );
        };
        /* first: place the main input if we don't double buffer */
        if (graph.nb_fields == 1) {
          insert_data(data);
        }
        /* second: place neighbor inputs */
        for (auto& ptr : cdata) {
          insert_data(ptr);
        }
      }

      /* TTG does not support thread IDs so we build our own */
      static thread_local int thread_id = -1;
      if (thread_id == -1) {
        static std::atomic<int> thread_cnt = 0;
        thread_id = thread_cnt++;
      }

      graph.execute_point(key.timestep, key.x, output_ptr, output_bytes,
                          input_ptrs, input_bytes, cdata.size()+1,
                          extra_local_memory[thread_id], graph.scratch_bytes_per_task);

      broadcast_keys(key, std::move(data), out);

    },
    // input edges
    ttg::edges(ttg::fuse(I2P, P2P),
               ttg::make_aggregator(ttg::fuse(P2C, I2C),
                                    [&](const Key& key){
                                      size_t numdeps = 0;
                                      TaskGraph& graph = graphs[key.graph_id];
                                      long last_offset = graph.offset_at_timestep(key.timestep-1);
                                      long last_width  = graph.width_at_timestep(key.timestep-1);
                                      for_each_dep(graph, key.x, key.timestep, false,
                                        [&](long x){
                                          // filter out dependencies that didn't exist in the previous timestep
                                          if (x < last_offset || x >= last_width) return false;
                                          if (graph.nb_fields == 1) {
                                            // we need to filter out the center point
                                            // sometimes it is included, sometimes it is not :/
                                            if (x != key.x) ++numdeps;
                                          } else {
                                            // double buffering: use all previous timestep's points as input */
                                            ++numdeps;
                                          }
                                          return false; // keep iterating
                                        }
                                      );
                                      //std::cout << "Aggregator " << key << " target " << numdeps << std::endl;
                                      return numdeps;
                                    })),
    // output edges
    ttg::edges(P2P, P2C, P2W), "Point", {"P2P", "Aggregator"}, {"P2P", "P2C", "P2W"}
  );

  /* if there is no graph with nb_field == 1 we can save space for a reduction in concurrency
   * by asking the tt to defer the next writer in the presence of readers,
   * avoiding a copy but delaying the execution of the writer until all readers completed.
   */
  bool all_graphs_nonunit_stride = std::end(graphs) == std::find_if(std::begin(graphs),
                                                                    std::end(graphs),
                                                                    [](const TaskGraph& graph){
                                                                      return graph.nb_fields == 1;
                                                                    });
  if (all_graphs_nonunit_stride) {
    point_tt->set_defer_writer(true);
  }

  point_tt->set_keymap(procmap);

  // TODO: this task should be inlined
  auto write_back_tt = ttg::make_tt(
    [&](const Key& key, data_t&& data){
      auto& graph_data = local_data[key.graph_id];
      graph_data(key.x, key.timestep) = std::move(data);
    }, ttg::edges(P2W), ttg::edges());

  write_back_tt->set_keymap(procmap);

  std::cout << "==== begin dot ====\n";
  std::cout << ttg::Dot()(init_tt.get()) << std::endl;
  std::cout << "====  end dot  ====\n";

  /* make sure the graph is connected */
  bool connected = ttg::make_graph_executable(init_tt);
  assert(connected);

  /* start the execution of tasks */
  ttg::execute();

  /* #### parsec context Starting #### */
  if (world.rank() == 0) {
    Timer::time_start();
  }

  /* invoke init on all processes */
  init_tt->invoke(Key(0, 0, 0));

  /* wait for all tasks to complete */
  ttg::fence();

  if (world.rank() == 0) {
    double elapsed = Timer::time_end();
    report_timing(elapsed);
  }

}

int main(int argc, char **argv)
{

  TTGApp app(argc, argv);
  app.execute_main_loop();

  return 0;
}

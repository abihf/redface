package thread

/*
   #define _GNU_SOURCE
   #include <sched.h>
   #include <pthread.h>

   void set_cpu_affinity(int core_id) {
       cpu_set_t cpuset;
       CPU_ZERO(&cpuset);
       CPU_SET(core_id, &cpuset);
       pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
   }
*/
import "C"

func SetCPUAffinity(coreID int) {
	C.set_cpu_affinity(C.int(coreID))
}

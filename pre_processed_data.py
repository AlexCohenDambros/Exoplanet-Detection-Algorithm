import time
import queue
from General_Functions import pre_processed_curves
from multiprocessing import Process, Manager, Queue, freeze_support

if __name__ == '__main__':
    
    num_threads = pre_processed_curves.multiprocessing.cpu_count()

    start_time = time.time()

    df_tess, df_kepler, df_k2 = pre_processed_curves.open_datasets(get_candidates= True)
    
    # telescopes_list = {'Kepler': df_kepler, 'TESS': df_tess}

    # TEST
    telescopes_list = {'Kepler': df_kepler.sample(10)}

    # ============= Execution of threads for data pre-processing =============
    local_curves = []
    global_curves = []
    local_global_target = []

    for name_telescope, df_telescope in telescopes_list.items():

        # Manager
        manager = Manager()

        # Flare gun
        finishedTheLines = manager.Event()

        # Processing Queues
        processinQqueue = Queue(df_telescope.shape[0])
        answerQueue = Queue(df_telescope.shape[0] + num_threads)

        threads = []

        for i in range(num_threads):
            threads.append(Process(target=pre_processed_curves.process_threads, args=(
                processinQqueue, answerQueue, finishedTheLines, name_telescope)))
            threads[-1].start()

        for _, row in df_telescope.iterrows():
            processinQqueue.put_nowait(row)

        time.sleep(1)
        finishedTheLines.set()

        threads_finished = 0
        while threads_finished < num_threads:
            try:
                get_result = answerQueue.get(False)
                if get_result == "ts":
                    threads_finished += 1
                    continue

                # Finish processing the data
                (target, data_local, data_global) = get_result
                local_global_target.append(target)
                local_curves.append(data_local)
                global_curves.append(data_global)

            except queue.Empty:
                continue

        for t in threads:
            t.join()

    # marks the end of the runtime
    end_time = time.time()

    # Calculates execution time in seconds
    execution_time = end_time - start_time

    print(f"Runtime: {execution_time:.2f} seconds")

    # Calls the function to save the preprocessed data locally
    pre_processed_curves.saving_preprocessed_data(local_curves, global_curves, local_global_target, candidate = False)
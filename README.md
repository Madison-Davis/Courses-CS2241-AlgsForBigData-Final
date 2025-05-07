Github Setup:

This Github allows for one to run and profile soft-actor critic (sac) methods in the Hopper v5 environment.  

The 'sac' folder has all of the sac files and other files related to profiling.

We have 4 main Soft-Actor Critic files that test variations in how their replay buffers (rb) work:
1. 'sac_torch_baseline'   sac + torch rl replay buffer (no prioritization)                    
2. 'sac_torch_prb'        sac + torch rl prioritized replay buffer (prioritization rb but no tiered cache)                
3. 'sac_pcrb'             sac + pcrb (prioritized and tiered cache replay buffer)
4. 'sac_pcrb_anneal'      sac + pcrb + parameter annealing         

To execute the files, simply run the requirements.txt to get all the necessary requirements, then run the file from the sac directory.
The results will show up in eval/runs.  To visualize them, type tensorboard --logdir=[the path to the directory with the runs] --port=16007 --host=localhost
Then go to localhost:6007

To profile the files, simply run the profiling equivalent file name of the sac algorithm you want to run.

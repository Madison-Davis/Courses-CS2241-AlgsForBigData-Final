Github Setup:

Github has 2 main files: sac and runs.

The 'sac' folder simply holds sac files (soft-actor-critic methods).
We have 3 main Soft-Actor Critic files that test variations in how their replay buffers (rb) work:
1. 'sac_baseline'   sac + stable baseline (no priorized rb)                         
2. 'sac_torch'      sac + torch (prioritized rb but no tiered cache)                
3. 'sac_pcrb'       sac + pcrb (prioritized and tiered cache rb)         

The first two files test replay buffers that were pre-written by other libraries.
The third sac file tests a customized file of a replay buffer based on pytorch.
This customized file is labeled 'prcb.py'.

The 'runs' folder holds the results for our tests on the sac files.  We will conduct 4 main tests.
We'll conduct the tests simply by executing the sac files in the CLI and specify arguments.
1. sac_baseline, timesteps = 1M
2. sac_torch, timesteps = 1M
3. sac_pcrb, timesteps = 1M, tiered cache sizes = [] vs [] vs [], quantization amt per tier = []
4. sac_pcrb, timesteps = 1M, tiered cache sizes = [], quantization amt per tier = [] vs [] vs []


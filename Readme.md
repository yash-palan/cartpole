Readme

Setup instructions
Please keep the relative structure of the file system the way it is on the github.

1. Training the RL system
The "train.py" file basically hosts the function of training the RL agent. The config inputs are inside the file (under the if __name__=="__main__" section). Here one can change the configuration to train the corresponding RL agent. 

The code will save all the following files
a. "ansatz_1_seed_4.pt" : Stores the network (torch nn.Module  inherited class)
b. "config_4.json" : Stores the configuration used for training
c. "rewards_per_episode_seed_4.npy" : Stores a numpy array containing the rewards of the agent per episode 
d. "train_rewards_summary_seed_4.jpg" : Plot showing the rewards per episode 

2. Extracting the compression ratio:
To extract the compression ratio, just run the function  "compression_percentage(....)" in the file "measure_of_usefulness.py". This is commented in the file and can be uncommented. 
Note- Change the relative file location to the correct "*.pt" file, inside "measure_of_usefulness.py" before running it.  

3. Extracting the Average entanglement ratio:
Same as compression ratio, with the difference that you need to run "extracting_and_infering_from_entanglement_entropy(...)".

4. Extracting the figure 3 and 4 in report:
To extract figure 3 and 4 in the report, once you have trained the necessary models, just open the "plotting_variance_plot.ipynb" and run the corresponding cells. 
Remember to change the argument of the functions "extaction_of_averaged_data_results(...)" to have the correct path. Also, look inside the function extaction_of_averaged_data_results in case the paths are different. 
Once the correct path is provided, the plots will be generated on their own. Note that this accesses the files 
"rewards_per_episode_seed_*.npy".

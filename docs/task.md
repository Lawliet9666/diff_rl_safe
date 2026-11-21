# Task 01: Understand the Repository and Background

## Goal
understand what this repository does and the purpose of its main components.
1. what is the 5 robot models and 5 methods at a high level.
2. Understand the roles of learning, robot dynamics, controllers, and safety filters without studying mathematical equations or derivations.
3. Find the main code/configuration folders and run one existing controller example. 

## Read First

**Note:** You can skip topics you already know. Focus on intuition and diagrams. You should be able to explain what each component does and why it is used; equations, proofs, and algorithm details are not required

1. **RL and PPO.** Understand the basic RL loop: try actions → collect experience and rewards → improve the policy → repeat. Read the “Key Concepts and Terminology” section of [RL basics](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html). For PPO, simply understand that it improves a policy using collected experience.
2. **Robot dynamics and controllers.** Dynamics describe how a robot moves in response to commands. A controller chooses commands using the robot's current state and goal. Watch [Control System Overview](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/) and focus on the feedback-loop diagram. You do not need to study the robot-arm equations or other chapters.
3. **CBF-QP.** Understand its purpose: a safety filter adjusts a proposed command using safety constraints, such as obstacle clearance and robot limits. A QP chooses an adjusted command close to the proposal. For BarrierNet, understand only that the safety layer participates in learning. The [CBF overview](https://arxiv.org/abs/1903.11199) and [BarrierNet paper](https://arxiv.org/abs/2111.11277) are optional references.

Watch the [README demo](../README.md), then skim Methods, Robots, Config, and Project Structure. Use these to connect the background to the repository.

## What to Do

1. **Understand the navigation task.** Identify the robot, goal, obstacles, and trajectory in the demo. Explain what reaching the goal, colliding, and running out of time look like.
2. **Compare the robots.** Make a five-row table describing what commands each robot accepts in plain language. For example, distinguish a command to move at a velocity from a command to accelerate. 
3. **Compare the methods.** For `cbf`, `ppo`, `ppo_filter`, `cbf_rl`, and `barriernet`, write one sentence explaining its purpose. State whether it learns a policy and when it uses a safety filter/layer: training, evaluation, or both. 
4. **Draw each method's architecture.** Create one flowchart for each of the five methods. Show the overall inputs, main modules, final robot command, and environment feedback. Label each module's purpose, inputs, and outputs, and show how training differs from evaluation. 
5. **Find three settings.** Locate the obstacle count, robot speed limit, and goal tolerance in the configuration files. Record their names, values, and file locations; you do not need to tune them.
6. **Test the unicycle CBF-QP controller.** Run 10 episodes using the default settings, then save one example video. No RL training is needed. Follow the README setup instructions and run from the repository root:

   ```bash
   # Check the settings without starting a simulation.
   CPU_THREADS=2 bash scripts/run_local.sh cbf unicycle --device cpu --cfg job

   # Collect results over 10 episodes (seeds 100 through 109).
   CPU_THREADS=2 bash scripts/run_local.sh cbf unicycle --device cpu \
     --seeds 100 --episodes 10

   # Save a video for one of the same test scenes.
   CPU_THREADS=2 bash scripts/run_local.sh cbf unicycle --device cpu \
     --seeds 100 --episodes 1 --visualize
   ```

   The 10-episode test and the video run create separate run directories under `outputs/controllers/`. Use `eval_results.json` from the **10-episode run** for the results table, and the MP4 under `visualize_*/` from the **one-episode run** for the video. Keep the saved `config.yaml` and report any errors or failures.

   Use these paper-style metrics:

   | Metric | Meaning | Field in `eval_results.json` |
   | --- | --- | --- |
   | SR (%) | Percentage of episodes reaching the goal. | `100 × success_rate` |
   | CR (%) | Percentage ending in collision. | `100 × collision_rate` |
   | TR (%) | Percentage ending at the time limit. | `100 × timeout_rate` |
   | Avg. Ret. | Average total reward per episode. | Currently not recorded for controllers; report **N/A**. |
   | Traj. Time (s) | Time to reach the goal, mean ± standard deviation over successful episodes. | `traj_time` ± `std_traj_time` |
   | Min. Dist. (m) | Each episode's closest robot–obstacle surface clearance, mean ± standard deviation over successful episodes. | `min_dist` ± `std_min_dist` |

   Read the controller's metric entry (`cbf_qp` or `cbf_qp_slack`), not the `aggregate` entry. If there are no successful episodes, report **N/A** for Traj. Time and Min. Dist. Also report `failure_rate` and `qp_failures` separately; SR, CR, and TR can sum to less than 100% when other terminal failures occur.

## Answer These Questions in the PR

Use short explanations in your own words. Mathematical derivations are not required.

1. **Purpose:** What problem does the repository solve? What are the roles of a learned policy, a controller, and a safety filter? 
2. **Robots and methods:** Include your robot and method tables. Explain the main differences in plain language.
3. **Method architecture flowcharts:** Draw **one flowchart for each method**: `cbf`, `ppo`, `ppo_filter`, `cbf_rl`, and `barriernet`. Each flowchart should explain how the whole method works:

   - Show the method's overall inputs, such as robot state, goal, and obstacle information, and the final command sent to the robot.
   - Draw the main modules as boxes. Label each box with **Input**, **Purpose**, and **Output**. Make clear what a learned policy produces and how that becomes the executed command.
   - Connect the boxes with arrows labeled by the information passed between them. Show how the environment produces the next state or observation and feeds it back into the method.
   - Show the differences between training and evaluation. For learning methods, include the reward and policy-update path, and mark which modules are used only during training.


4. **Controller test results:** Report your 10-episode unicycle test in this table, filling in the measured values:

   | Robot / Controller | Episodes | SR (%) | CR (%) | TR (%) | Avg. Ret. | Traj. Time (s) | Min. Dist. (m) |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | Robot Type / CBF-QP | 10 | … | … | … | N/A | … ± … | … ± … |

   Link one test video and briefly explain how the robot approaches the goal and avoids obstacles, or why the episode fails.

## Submit Your Work

Fork the repository, work in a development branch, and open a PR into your fork's `main`. Put your answers, diagrams, commands, and result/video links in the PR. Send the PR link to the instructor.

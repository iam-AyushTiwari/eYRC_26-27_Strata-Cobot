# e-Yantra Robotics Competition 2026-27 — StrataCobot (SC)

Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic

The task pages are the full instructions. This page is the short version.

## 1. Install and build

From this directory, with ROS 2 Jazzy already installed:

```bash
./requirements.sh
cd ..
colcon build
source install/setup.bash
```

## 2. Start a world

Each subtask has its own launch file. Leave it running in its own terminal.

```bash
ros2 launch eyantra_kepler_colony task3a.launch.py   # Task 3A
ros2 launch eyantra_kepler_colony task3b.launch.py   # Task 3B
```

Wait for `Task 3A world ready` or `Task 3B world ready` before going on. For an evaluation,
start the evaluator first (step 4).

## 3. Your scripts

Your code goes in the **`algorithms`** package, under these names:

| Subtask | Folder | Scripts |
|---|---|---|
| 3A | `algorithms/scripts/task3a/` | `task3A_perception.py`, `task3A_manipulation.py`, `task3A_planning.py`, `task3A_navigation.py` |
| 3B | `algorithms/scripts/task3b/` | `task3B_perception.py`, `task3B_manipulation.py`, `task3B_planning.py`, `task3B_navigation.py` |

For each script: make it executable (`chmod +x`), add its path to `SCRIPTS` in
`algorithms/setup.py` (e.g. `'scripts/task3a/task3A_perception.py'`), then rebuild and run it:

```bash
colcon build --packages-select algorithms
source install/setup.bash
ros2 run algorithms task3A_perception.py
```

Rebuild and re-source after every edit: `ros2 run` starts the installed copy.

A launch file of your own that starts all four also works. Put it in `algorithms/launch/`, which
`setup.py` installs, or in a package of your own. It starts your nodes only: launch the world on
its own, as in step 2.

## 4. Evaluate and submit

Run the evaluator from this directory, in a terminal where the workspace is sourced
(`source ../install/setup.bash`). It starts **before** the world.

Task 3A:

1. `./eyrc-sc-evaluator --task 3A --team-id <YOUR_TEAM_ID>`. It waits for the world.
2. In another terminal, start the world as in step 2.
3. When the evaluator prints `Ready`, start your four scripts.
4. Press `q` in the evaluator when both robots have finished; a run that earns the bonus ends
   on its own. It writes `result.zip`. Do not unpack or edit it.
5. Zip `result.zip` with your four scripts, any other node you ran (named
   `task3A_<what_it_does>.py`), and your launch file if you used one (named
   `task3A_nodes.launch.py`), and upload it, named as the submission page says.

Task 3B is two runs in one evaluation, in the same order:
`./eyrc-sc-evaluator --task 3B --team-id <YOUR_TEAM_ID>`, then the world, then your scripts at
`Ready`. Press `x` to end the first run (a run that earns the bonus ends on its own). Stop the
scripts and the world, launch the world again within 3 minutes, start the same scripts at
`Ready`, and press `q` to end the second run. The submission page has every step.

---

The UR7e description in `ur_description` is modified by e-Yantra from the upstream
Universal Robots ROS 2 description; see `ur_description/LICENSE`.

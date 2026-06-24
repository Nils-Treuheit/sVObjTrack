What you are experiencing is a notorious ROS 2 quirk: **`colcon` hardcodes absolute python paths into the generated executable entry points.**

When you run `colcon build`, `setuptools` looks at how `colcon` itself was called. If your `colcon` was installed via `apt` (system-wide), it forcefully injects `#!/usr/bin/python3` into the top of the generated wrapper scripts inside your `install/` directory. No amount of environment sourcing can change a hardcoded path.

To bypass this and break the system-wide chain, you have to do two things to make `colcon` generate dynamic, environment-aware scripts.

---

### Step 1: Force a dynamic interpreter via `setup.cfg`

Open the `setup.cfg` file inside your ROS 2 Python package(s) and add (or modify) the `[build_scripts]` section to look exactly like this:

```ini
[build_scripts]
executable = /usr/bin/env python3

```

> **Why this works:** By default, `colcon` forces the absolute path of the system interpreter (`/usr/bin/python3`). By explicitly mapping the executable to `/usr/bin/env python3`, you are telling the system to look at whatever `python3` is **currently active in your terminal** (which will be your `uv` `.venv`).

---

### Step 2: Use `python3 -m colcon` instead of the standalone `colcon` command

Because the global `colcon` binary has a fixed system shebang, it tends to ignore virtual environments when generating build artifacts.

Nuke your previous build attempts and build using your local `uv` environment's Python explicitly invoking the colcon module:

```bash
# 1. Clear out the bad builds
rm -rf build/ install/ log/

# 2. Make sure your uv venv is active
source .venv/bin/activate

# 3. Build by running colcon directly THROUGH your venv's python module
colcon build --cmake-args -DPYTHON_EXECUTABLE=$(which python) --symlink-install

```

---

### Step 3: Run the Node (Mind the Order!)

Open a fresh terminal, and make sure your environment variables layer over each other correctly:

```bash
# 1. System ROS 2 underlying paths
source /opt/ros/<distro>/setup.bash

# 2. Activate uv environment (Puts your venv python at the front of $PATH)
source .venv/bin/activate

# 3. Source your newly compiled workspace
source install/setup.bash

# 4. Run your node
ros2 run my_package my_node

```

Check your `sys.executable` printout now—it should successfully display your local project `.venv/bin/python3`!
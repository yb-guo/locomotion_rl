# 002: Genesis Franka Effort API Smoke

## Route

Build the Genesis Franka scene and verify the effort/current proxy signals.

Preferred first asset:

```text
xml/franka_emika_panda/panda_nohand.xml
```

Load with Jacobian/IK enabled:

```text
requires_jac_and_IK=True
```

Record whether these APIs exist and produce finite values:

```text
get_dofs_position
get_dofs_velocity
get_dofs_control_force
get_dofs_force
get_jacobian
```

Official API interpretation to verify and record:

- `get_dofs_control_force()` returns controller-applied force/torque. Under
  position or velocity control it is computed from target command and gains.
  Under force control it equals the input force command.
- `get_dofs_force()` returns actual internal DOF force at the current timestep
  and may include controller force plus internal effects such as collision and
  Coriolis terms.
- `get_jacobian(link, local_point=None)` returns a spatial Jacobian with shape
  `(6, n_dofs)` or `(n_envs, 6, n_dofs)`.

The smoke must identify:

- arm joint names and local DOF indices;
- selected tool/end-effector link name;
- whether Jacobian rows `0:3` are translational and rows `3:6` rotational in
  the installed Genesis version;
- whether `panda_nohand.xml` has a suitable terminal link for payload
  attachment.

Task-local script:

```text
.agent/task/task023-franka-current-payload-estimation/genesis_franka_effort_api_smoke.py
```

The script must stay task-local until feasibility passes. It should write a JSON
summary under:

```text
outputs/task023/franka_current_force_estimation/
```

## Log

- 2026-05-13 Created in the task023 replanning pass.
- 2026-05-13 Added official API semantics for control force, internal force,
  and Jacobian, plus the requirement to enable Jacobian/IK support.
- 2026-05-13 Added task-local smoke script. It lazy-imports Genesis, requests
  `requires_jac_and_IK=True` for `panda_nohand`, probes
  `get_dofs_position`, `get_dofs_velocity`, `get_dofs_control_force`,
  `get_dofs_force`, `get_jacobian`, tool-link resolution, and weld API
  availability.
- 2026-05-13 Syntax check without writing pyc:
  `python -c "import ast,pathlib; ast.parse(pathlib.Path('.agent/task/task023-franka-current-payload-estimation/genesis_franka_effort_api_smoke.py').read_text(encoding='utf-8')); print('AST_OK')"`
  -> `AST_OK`.
- 2026-05-13 Local smoke command:
  `python .agent/task/task023-franka-current-payload-estimation/genesis_franka_effort_api_smoke.py --backend cpu --output outputs/task023/franka_current_force_estimation/local_genesis_api_smoke.json`
  -> `status=blocked`,
  `blocker=genesis_import_failed:No module named 'genesis'`.
- 2026-05-13 `python -m py_compile ...` was not used as validation because
  Windows denied writing the generated pyc file under task-local
  `__pycache__`; the AST parse above is the syntax evidence.
- 2026-05-13 Synced the task-local smoke script to H200:
  `/root/agent_workspace/project/h200-locomotion-lab-task023-franka-current-payload-estimation/.agent/task/task023-franka-current-payload-estimation/genesis_franka_effort_api_smoke.py`.
- 2026-05-13 H200 guarded command:
  `C:\Windows\System32\OpenSSH\ssh.exe myserver "/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task023-franka-current-payload-estimation && CUDA_VISIBLE_DEVICES=1 python .agent/task/task023-franka-current-payload-estimation/genesis_franka_effort_api_smoke.py --backend cuda --output outputs/task023/franka_current_force_estimation/gpu_genesis_api_smoke.json'"`
  -> `status=ok`.
- 2026-05-13 H200 evidence:
  `outputs/task023/franka_current_force_estimation/gpu_genesis_api_smoke.json`
  copied back locally. Summary:
  - Genesis `0.4.6`, CUDA backend, Python `3.11.11`, Linux H200 target.
  - Asset `xml/franka_emika_panda/panda_nohand.xml`.
  - `robot_n_dofs=7`.
  - Arm joints `joint1..joint7`, local DOFs `[0,1,2,3,4,5,6]`.
  - Tool link resolved as `link7`, index `8`.
  - `requires_jac_and_IK=True` requested and used.
  - `get_dofs_control_force`: finite, shape `[1,7]`.
  - `get_dofs_force`: finite, shape `[1,7]`.
  - `get_dofs_position`: finite, shape `[1,7]`.
  - `get_dofs_velocity`: finite, shape `[1,7]`.
  - `get_jacobian`: finite, shape `[1,6,7]`, translational row norms
    `[0.7984514528427935, 0.1652012039064354, 0.1314249156140556]`.
  - Weld probe: `add_weld_constraint_available=true`,
    `delete_weld_constraint_available=true`, `smoke_status=ok`.
  - Warnings observed: Genesis reported `torch<2.8.0` unsupported and neutral
    robot `qpos0` exceeding joint limits. These do not block the API smoke but
    must be considered in trajectory initialization.

## Review

Status: passed for H200 API smoke.

Evidence:

- Task-local smoke script exists.
- Local syntax check passed.
- Local execution wrote
  `outputs/task023/franka_current_force_estimation/local_genesis_api_smoke.json`
  with a clean blocked reason.
- H200 execution wrote
  `outputs/task023/franka_current_force_estimation/gpu_genesis_api_smoke.json`
  with finite effort, state, Jacobian, and weld API results.

Next evidence needed:

- Subtask `003` should use the verified H200 APIs to generate slow
  step-aligned payload traces. Avoid assuming the default `qpos0` is a good
  trajectory seed because the smoke emitted a neutral-position joint-limit
  warning.

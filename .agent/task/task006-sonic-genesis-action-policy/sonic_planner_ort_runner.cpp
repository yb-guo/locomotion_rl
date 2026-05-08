#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kQposDim = 36;
constexpr int kMotorDim = 29;
constexpr int kContextFrames = 4;
constexpr float kDefaultHeight = 0.788740f;

struct Args {
  std::string planner;
  std::string initial_joint_pos_csv;
  int initial_joint_pos_row = 0;
  std::string output_qpos_csv;
  int mode = 2;
  float target_vel = -1.0f;
  std::array<float, 3> movement_direction{1.0f, 0.0f, 0.0f};
  std::array<float, 3> facing_direction{1.0f, 0.0f, 0.0f};
  int64_t random_seed = 1234;
  float height = -1.0f;
};

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream stream(line);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

std::vector<float> ReadJointRow(const std::string& path, int row_index) {
  if (path.empty()) {
    return {
        -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f,
        0.669f, -0.363f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.0f,
        0.6f, 0.0f, 0.0f, 0.0f, 0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f,
        0.0f};
  }

  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("failed to open initial joint CSV: " + path);
  }
  std::string line;
  int numeric_row = 0;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    auto fields = SplitCsvLine(line);
    if (fields.empty()) {
      continue;
    }
    std::vector<float> values;
    try {
      for (const auto& field : fields) {
        values.push_back(std::stof(field));
      }
    } catch (const std::exception&) {
      if (numeric_row == 0) {
        continue;
      }
      throw;
    }
    if (values.size() != kMotorDim) {
      throw std::runtime_error("expected 29 joint values, got " + std::to_string(values.size()));
    }
    if (numeric_row == row_index) {
      return values;
    }
    numeric_row += 1;
  }
  throw std::runtime_error("initial joint CSV row not found: " + std::to_string(row_index));
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (key == "--planner") {
      args.planner = require_value(key);
    } else if (key == "--initial-joint-pos-csv") {
      args.initial_joint_pos_csv = require_value(key);
    } else if (key == "--initial-joint-pos-row") {
      args.initial_joint_pos_row = std::stoi(require_value(key));
    } else if (key == "--output-qpos-csv") {
      args.output_qpos_csv = require_value(key);
    } else if (key == "--mode") {
      args.mode = std::stoi(require_value(key));
    } else if (key == "--target-vel") {
      args.target_vel = std::stof(require_value(key));
    } else if (key == "--height") {
      args.height = std::stof(require_value(key));
    } else if (key == "--random-seed") {
      args.random_seed = std::stoll(require_value(key));
    } else if (key == "--movement-direction") {
      args.movement_direction = {
          std::stof(require_value(key)),
          std::stof(require_value(key)),
          std::stof(require_value(key))};
    } else if (key == "--facing-direction") {
      args.facing_direction = {
          std::stof(require_value(key)),
          std::stof(require_value(key)),
          std::stof(require_value(key))};
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.planner.empty()) {
    throw std::runtime_error("--planner is required");
  }
  if (args.output_qpos_csv.empty()) {
    throw std::runtime_error("--output-qpos-csv is required");
  }
  return args;
}

std::vector<float> BuildContext(const std::vector<float>& joints) {
  std::vector<float> context(kContextFrames * kQposDim, 0.0f);
  for (int frame = 0; frame < kContextFrames; ++frame) {
    int base = frame * kQposDim;
    context[base + 0] = 0.0f;
    context[base + 1] = 0.0f;
    context[base + 2] = kDefaultHeight;
    context[base + 3] = 1.0f;
    context[base + 4] = 0.0f;
    context[base + 5] = 0.0f;
    context[base + 6] = 0.0f;
    std::copy(joints.begin(), joints.end(), context.begin() + base + 7);
  }
  return context;
}

template <typename T>
Ort::Value Tensor(
    Ort::MemoryInfo& memory_info,
    std::vector<T>& values,
    const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor<T>(
      memory_info,
      values.data(),
      values.size(),
      shape.data(),
      shape.size());
}

bool AllFinite(const float* values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

void WriteQposCsv(const std::string& path, const float* values, int rows) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open output qpos CSV: " + path);
  }
  out.setf(std::ios::fixed);
  out.precision(9);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < kQposDim; ++col) {
      if (col > 0) {
        out << ",";
      }
      out << values[row * kQposDim + col];
    }
    out << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);
    auto joints = ReadJointRow(args.initial_joint_pos_csv, args.initial_joint_pos_row);
    auto context = BuildContext(joints);

    std::vector<float> target_vel{args.target_vel};
    std::vector<int64_t> mode{args.mode};
    std::vector<float> movement_direction{
        args.movement_direction[0], args.movement_direction[1], args.movement_direction[2]};
    std::vector<float> facing_direction{
        args.facing_direction[0], args.facing_direction[1], args.facing_direction[2]};
    std::vector<int64_t> random_seed{args.random_seed};
    std::vector<int64_t> has_specific_target{0};
    std::vector<float> specific_target_positions(12, 0.0f);
    std::vector<float> specific_target_headings(4, 0.0f);
    std::vector<int64_t> allowed_pred_num_tokens{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    std::vector<float> height{args.height};

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sonic_planner_ort_runner");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    Ort::Session session(env, args.planner.c_str(), options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.push_back(Tensor(memory_info, context, {1, 4, 36}));
    inputs.push_back(Tensor(memory_info, target_vel, {1}));
    inputs.push_back(Tensor(memory_info, mode, {1}));
    inputs.push_back(Tensor(memory_info, movement_direction, {1, 3}));
    inputs.push_back(Tensor(memory_info, facing_direction, {1, 3}));
    inputs.push_back(Tensor(memory_info, random_seed, {1}));
    inputs.push_back(Tensor(memory_info, has_specific_target, {1, 1}));
    inputs.push_back(Tensor(memory_info, specific_target_positions, {1, 4, 3}));
    inputs.push_back(Tensor(memory_info, specific_target_headings, {1, 4}));
    inputs.push_back(Tensor(memory_info, allowed_pred_num_tokens, {1, 11}));
    inputs.push_back(Tensor(memory_info, height, {1}));

    const char* input_names[] = {
        "context_mujoco_qpos",
        "target_vel",
        "mode",
        "movement_direction",
        "facing_direction",
        "random_seed",
        "has_specific_target",
        "specific_target_positions",
        "specific_target_headings",
        "allowed_pred_num_tokens",
        "height"};
    const char* output_names[] = {"mujoco_qpos", "num_pred_frames"};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        2);

    const float* qpos = outputs[0].GetTensorData<float>();
    const int32_t* num_pred_frames = outputs[1].GetTensorData<int32_t>();
    int rows = static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
    int cols = static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2]);
    if (cols != kQposDim) {
      throw std::runtime_error("planner qpos cols mismatch");
    }
    int pred_rows = std::max(0, std::min(rows, static_cast<int>(num_pred_frames[0])));
    if (pred_rows == 0) {
      throw std::runtime_error("planner returned zero predicted frames");
    }

    WriteQposCsv(args.output_qpos_csv, qpos, rows);

    float min_z = qpos[2];
    float max_z = qpos[2];
    for (int row = 0; row < pred_rows; ++row) {
      min_z = std::min(min_z, qpos[row * kQposDim + 2]);
      max_z = std::max(max_z, qpos[row * kQposDim + 2]);
    }

    std::cout << "SONIC_PLANNER_ORT_RUNNER_MODE onnxruntime_cpp\n";
    std::cout << "PLANNER " << args.planner << "\n";
    std::cout << "OUTPUT_QPOS_CSV " << args.output_qpos_csv << "\n";
    std::cout << "PLANNER_MODE " << args.mode << "\n";
    std::cout << "TARGET_VEL " << args.target_vel << "\n";
    std::cout << "PLANNER_QPOS_ROWS " << rows << "\n";
    std::cout << "PLANNER_QPOS_COLS " << cols << "\n";
    std::cout << "PLANNER_NUM_PRED_FRAMES " << num_pred_frames[0] << "\n";
    std::cout << "PLANNER_QPOS_FINITE " << AllFinite(qpos, static_cast<size_t>(rows * cols)) << "\n";
    std::cout << "PLANNER_ROOT_Z_MIN_MAX " << min_z << " " << max_z << "\n";
    std::cout << "SONIC_PLANNER_ORT_RUNNER_OK\n";
  } catch (const std::exception& exc) {
    std::cerr << "SONIC_PLANNER_ORT_RUNNER_ERROR " << exc.what() << "\n";
    return 1;
  }
  return 0;
}

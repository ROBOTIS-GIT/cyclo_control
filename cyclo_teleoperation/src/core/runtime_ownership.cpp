// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "cyclo_teleoperation/core/runtime_ownership.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <cstdint>
#include <stdexcept>

namespace cyclo_teleoperation
{
RuntimeOwnership::RuntimeOwnership(rclcpp::Node & node, const std::string & source_topic)
: node_(node),
  topic_(node.get_node_topics_interface()->resolve_topic_name(source_topic)),
  start_(std::chrono::steady_clock::now())
{
  size_t domain = 0;
  if (rcl_context_get_domain_id(node.get_node_base_interface()->get_context()->
    get_rcl_context().get(), &domain) != RCL_RET_OK)
  {
    throw std::runtime_error("Cannot determine runtime ownership domain");
  }
  uint64_t key = 14695981039346656037ULL;
  for (const unsigned char character : topic_) {
    key = (key ^ character) * 1099511628211ULL;
  }
  const auto path = "/tmp/cyclo-control-" + std::to_string(domain) + "-" +
    std::to_string(key) + ".lock";
  lock_fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (lock_fd_ < 0) {
    throw std::runtime_error("Cannot open runtime ownership lock: " + path);
  }
  if (flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
    close(lock_fd_);
    lock_fd_ = -1;
    throw std::runtime_error("Another control runtime already owns " + topic_);
  }
}

RuntimeOwnership::~RuntimeOwnership()
{
  if (lock_fd_ >= 0) {close(lock_fd_);}
  // Do not unlink the lock: a replacement inode would let concurrent owners through.
}

void RuntimeOwnership::check()
{
  const auto count = node_.count_publishers(topic_);
  if (count > 1) {
    ready_ = false;
    throw std::runtime_error("Multiple control runtimes detected on " + topic_);
  }
  ready_ = count == 1 &&
    std::chrono::steady_clock::now() - start_ >= std::chrono::seconds(1);
}
}  // namespace cyclo_teleoperation

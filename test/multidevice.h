// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <multidevice/communicator.h>
#include <test/utils.h>

namespace nvfuser {

class MultiDeviceTest : public NVFuserTest {
  public:
    Communicator& get_communicator() {
      return *comm;
    }
  protected:
    static void SetUpTestSuite() {
      std::cout << "NCCL communicator create" << std::endl;
      comm = new Communicator(CommunicatorBackend::nccl);
    }
    static void TearDownTestSuite() {
      delete comm;
      comm = nullptr;
    }
    static Communicator* comm;
};

class UCCMultiDeviceTest : public NVFuserTest {
  public:
    Communicator& get_communicator() {
      return *ucomm;
    }
  protected:
    static void SetUpTestSuite() {
      std::cout << "Init UCC communicator" << std::endl;
      ucomm = new Communicator(CommunicatorBackend::ucc);
      std::cout << "Communicator created" << std::endl;
    }
    static void TearDownTestSuite() {
      std::cout << "Tear down UCC communicator" << std::endl;
      delete ucomm;
      ucomm = nullptr;
    }
    static Communicator* ucomm;
};

} // namespace nvfuser

#endif

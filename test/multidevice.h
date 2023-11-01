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
      comm = new Communicator(CommunicatorBackendType::nccl);
    }
    static void TearDownTestSuite() {
      delete comm;
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
      ucomm = new Communicator(CommunicatorBackendType::ucc);
    }
    static void TearDownTestSuite() {
	    std::cout << "going to delete ucc communicator" << std::endl;
	    delete ucomm;
	    std::cout << "deleted ucc communicator" << std::endl;
    }
    static Communicator* ucomm;
};

} // namespace nvfuser

#endif

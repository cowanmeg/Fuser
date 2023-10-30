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
      comm = new Communicator(CommunicatorBackend::nccl, 0, 5001);
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
      ucomm = new Communicator(CommunicatorBackend::ucc, 0, 5002);
    }
    static void TearDownTestSuite() {
	    std::cout << "delete ucc communicator" << std::endl;
	    ucomm->barrier();
	    delete ucomm;
    }
    static Communicator* ucomm;
};

} // namespace nvfuser

#endif

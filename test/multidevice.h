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
      ucomm = new Communicator(CommunicatorBackend::ucc);
    }
    static void TearDownTestSuite() {
      if (ucomm) {
        delete ucomm;
        ucomm = nullptr;
      }
    }
    static Communicator* ucomm;
};

} // namespace nvfuser

#endif

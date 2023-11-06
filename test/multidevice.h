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
      // comm = std::make_unique<Communicator>(CommunicatorBackendType::nccl);
    }
    static void TearDownTestSuite() {
	    comm->barrier();
    	std::cout << "Teardown" << std::endl;
    }
    //static std::unique_ptr<Communicator> comm;
    static Communicator* comm;
};

class UCCMultiDeviceTest : public MultiDeviceTest {

  protected:
    static void SetUpTestSuite() {
	    if (comm == nullptr)
		   comm = new Communicator(CommunicatorBackendType::ucc);
	    else {
		    std::cout << "Reuse same communicator" << std::endl;
		    comm->addBackend(CommunicatorBackendType::ucc);
				comm->makeDefaultBackend(CommunicatorBackendType::ucc);
			}	
     // comm = std::make_unique<Communicator>(CommunicatorBackendType::ucc);
    }
    static void TearDownTestSuite() {
			comm->barrier();
			std::cout << "Tear down tst suit. Will delete ucc communicator" << std::endl;
    }
    //static std::unique_ptr<Communicator> comm;
};

} // namespace nvfuser

#endif

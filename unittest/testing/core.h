/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#ifndef TESTING_CORE__H
#define TESTING_CORE__H

#include "exceptions.h"
#include "info.h"
#include "input.h"
#include "lammps.h"
#include "platform.h"
#include "variable.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <functional>
#include <string>
#include <vector>

using LAMMPS_NS::Info;
using LAMMPS_NS::LAMMPS;
using LAMMPS_NS::LAMMPSException;

using ::testing::ContainsRegex;

#define TEST_FAILURE(errmsg, ...)                             \
    {                                                         \
        ::testing::internal::CaptureStdout();                 \
        ASSERT_ANY_THROW({__VA_ARGS__});                      \
        auto mesg = ::testing::internal::GetCapturedStdout(); \
        ASSERT_THAT(mesg, ContainsRegex(errmsg));             \
    }

// whether to print verbose output (i.e. not capturing LAMMPS screen output).
extern bool verbose;

class LAMMPSTest : public ::testing::Test {
public:
    void command(const std::string &line) { lmp->input->one(line); }

    void BEGIN_HIDE_OUTPUT()
    {
        if (!verbose) ::testing::internal::CaptureStdout();
    }

    void END_HIDE_OUTPUT()
    {
        if (!verbose) ::testing::internal::GetCapturedStdout();
    }

    void BEGIN_CAPTURE_OUTPUT() { ::testing::internal::CaptureStdout(); }

    std::string END_CAPTURE_OUTPUT()
    {
        auto output = ::testing::internal::GetCapturedStdout();
        if (verbose) std::cout << output;
        return output;
    }

    void HIDE_OUTPUT(std::function<void()> f)
    {
        if (!verbose) ::testing::internal::CaptureStdout();
        try {
            f();
        } catch (LAMMPSException &e) {
            if (!verbose) std::cout << ::testing::internal::GetCapturedStdout();
            throw e;
        }
        if (!verbose) ::testing::internal::GetCapturedStdout();
    }

    std::string CAPTURE_OUTPUT(std::function<void()> f)
    {
        ::testing::internal::CaptureStdout();
        try {
            f();
        } catch (LAMMPSException &e) {
            if (verbose) std::cout << ::testing::internal::GetCapturedStdout();
            throw e;
        }
        auto output = ::testing::internal::GetCapturedStdout();
        if (verbose) std::cout << output;
        return output;
    }

    double get_variable_value(const std::string &name)
    {
        char *str    = LAMMPS_NS::utils::strdup(fmt::format("v_{}", name));
        double value = lmp->input->variable->compute_equal(str);
        delete[] str;
        return value;
    }

    std::string get_variable_string(const std::string &name)
    {
        return lmp->input->variable->retrieve(name.c_str());
    }

protected:
    std::string testbinary = "LAMMPSTest";
    LAMMPS::argv args      = {"-log", "none", "-echo", "screen", "-nocite"};
    LAMMPS *lmp;
    Info *info;

    void SetUp() override
    {
        LAMMPS::argv full_args = {testbinary};
        full_args.insert(full_args.end(), args.begin(), args.end());

        HIDE_OUTPUT([&] {
            lmp  = new LAMMPS(full_args, MPI_COMM_WORLD);
            info = new Info(lmp);
        });
        InitSystem();
    }

    virtual void InitSystem() {}

    void TearDown() override
    {
        HIDE_OUTPUT([&] {
            delete info;
            delete lmp;
            info = nullptr;
            lmp  = nullptr;
        });
        std::cout.flush();
    }
};

#endif

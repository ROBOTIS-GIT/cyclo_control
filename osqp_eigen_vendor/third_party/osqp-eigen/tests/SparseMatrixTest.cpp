// Copyright (C) 2020 Giulio Romualdi
//
// Use of this source code is governed by a BSD-3-Clause
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/BSD-3-Claused

/**
 * @file SparseMatrixTest.cpp
 * @author Giulio Romualdi
 * @copyright  Released under the terms of the BSD 3-Clause License
 * @date 2020
 */

#include <osqp.h>

#include <iostream>

#include <OsqpEigen/OsqpEigen.h>  // NOLINT(build/include_order)

// Catch2
#include <catch2/catch_test_macros.hpp>

template<typename T, int n, int m> bool computeTest(const Eigen::Matrix<T, n, m> & mEigen)
{
  Eigen::SparseMatrix<T, Eigen::ColMajor> matrix, newMatrix, newMatrixFromCSR;
  matrix = mEigen.sparseView();

  csc * osqpSparseMatrix = nullptr;
  // NOTE: Dynamic memory allocation
  if (!OsqpEigen::SparseMatrixHelper::createOsqpSparseMatrix(matrix, osqpSparseMatrix)) {
    return false;
  }

  Eigen::SparseMatrix<T, Eigen::RowMajor> csrMatrix;
  csrMatrix = matrix;
  csc * otherOsqpSparseMatrix = nullptr;
  if (!OsqpEigen::SparseMatrixHelper::createOsqpSparseMatrix(csrMatrix, otherOsqpSparseMatrix)) {
    return false;
  }

  if (!OsqpEigen::SparseMatrixHelper::osqpSparseMatrixToEigenSparseMatrix(osqpSparseMatrix,
                                                                            newMatrix))
  {
    return false;
  }

  if (!OsqpEigen::SparseMatrixHelper::osqpSparseMatrixToEigenSparseMatrix(otherOsqpSparseMatrix,
                                                                            newMatrixFromCSR))
  {
    return false;
  }

  if (!newMatrixFromCSR.isApprox(newMatrix)) {
    return false;
  }

  std::vector<Eigen::Triplet<T>> tripletListCsc;
  if (!OsqpEigen::SparseMatrixHelper::osqpSparseMatrixToTriplets(osqpSparseMatrix,
                                                                   tripletListCsc))
  {
    return false;
  }

  for (const auto & a : tripletListCsc) {
    std::cout << a.row() << " " << a.col() << " " << a.value() << std::endl;
  }

  std::vector<Eigen::Triplet<T>> tripletListEigen;
  OsqpEigen::SparseMatrixHelper::eigenSparseMatrixToTriplets(matrix, tripletListEigen);

  std::cout << "***********************************************" << std::endl;
  for (const auto & a : tripletListEigen) {
    std::cout << a.row() << " " << a.col() << " " << a.value() << std::endl;
  }

  constexpr double tolerance = 1e-4;
  bool outcome = matrix.isApprox(newMatrix, tolerance);

  OsqpEigen::spfree(osqpSparseMatrix);
  OsqpEigen::spfree(otherOsqpSparseMatrix);

  return outcome;
}

TEST_CASE("SparseMatrix")
{
  SECTION("Data type - double")
  {
    Eigen::Matrix3d m;
    m << 0, 1.002311, 0, 0, 0, 0, 0, 0.90835435, 0;

    REQUIRE(computeTest(m));
  }

  SECTION("Data type - float")
  {
    Eigen::Matrix3f m;
    m << 0, 1, 0, 0, 0, 0, 0, 1, 0;

    REQUIRE(computeTest(m));
  }

  SECTION("Data type - int")
  {
    Eigen::Matrix3i m;
    m << 0, 1, 0, 0, 0, 0, 0, 1, 0;

    REQUIRE(computeTest(m));
  }

  SECTION("Data type - double")
  {
    Eigen::Matrix<double, 4, 2> m;
    m << 0, 0, 0, 4, 0, 0, 0, 0;

    REQUIRE(computeTest(m));
  }
}

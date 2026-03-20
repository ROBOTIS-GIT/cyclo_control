// Copyright (C) 2023 Giulio Romualdi
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

// Copyright 2023 Giulio Romualdi

#include <osqp_api_functions.h>

int main()
{
  // This function is only available in OSQP >= 1.0.0
  OSQPCscMatrix_set_data(nullptr, 0, 0, 0, nullptr, 0, 0);
  return 0;
}

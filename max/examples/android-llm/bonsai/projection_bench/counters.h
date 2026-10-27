// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt Optional calibrated KGSL capture. The pinned
// UAPI header is supplied externally.
#pragma once
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#define __user
#include "hill-msm_kgsl.h"

class ProjectionCounters {
  struct Entry {
    unsigned group, selector;
    unsigned long long before = 0;
    bool active = false;
  };
  std::array<Entry, 7> entries{
      {{13, 34}, {13, 35}, {13, 46}, {13, 47}, {10, 100}, {10, 8}, {10, 9}}};
  int fd = -1;
  void require(bool ok, const char *operation) {
    if (!ok) {
      fprintf(stderr, "KGSL %s failed: %s\n", operation, strerror(errno));
      exit(2);
    }
  }

public:
  ProjectionCounters() {
    fd = open("/dev/kgsl-3d0", O_RDWR);
    require(fd >= 0, "open");
    for (auto &entry : entries) {
      kgsl_perfcounter_get get{};
      get.groupid = entry.group;
      get.countable = entry.selector;
      require(ioctl(fd, IOCTL_KGSL_PERFCOUNTER_GET, &get) == 0, "get");
      entry.active = true;
    }
  }
  void snapshot(bool before, unsigned long long expected = 0) {
    std::array<kgsl_perfcounter_read_group, 7> reads{};
    for (size_t i = 0; i < entries.size(); ++i) {
      reads[i].groupid = entries[i].group;
      reads[i].countable = entries[i].selector;
    }
    kgsl_perfcounter_read request{};
    request.reads = reads.data();
    request.count = reads.size();
    require(ioctl(fd, IOCTL_KGSL_PERFCOUNTER_READ, &request) == 0, "read");
    std::array<unsigned long long, 7> delta{};
    for (size_t i = 0; i < entries.size(); ++i) {
      if (before)
        entries[i].before = reads[i].value;
      else
        delta[i] = reads[i].value - entries[i].before;
    }
    if (!before)
      printf("{\"stage\":\"counters\",\"read_bytes\":%llu,\"write_bytes\":%llu,"
             "\"launched_threads\":%llu,\"expected_threads\":%llu,\"valid_"
             "invocations\":%s,\"resident_context_numerator\":%llu,\"resident_"
             "context_denominator\":%llu}\n",
             32 * (delta[0] + delta[1]), 32 * (delta[2] + delta[3]), delta[4],
             expected, delta[4] == expected ? "true" : "false", delta[5],
             delta[6]);
  }
  ~ProjectionCounters() {
    for (auto &entry : entries)
      if (entry.active) {
        kgsl_perfcounter_put put{};
        put.groupid = entry.group;
        put.countable = entry.selector;
        ioctl(fd, IOCTL_KGSL_PERFCOUNTER_PUT, &put);
      }
    if (fd >= 0)
      close(fd);
  }
};

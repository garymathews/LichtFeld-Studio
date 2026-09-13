/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <streambuf>
namespace lfs::training {
        class CountingStreamBuffer final
            : public std::streambuf {
        public:
            [[nodiscard]] std::uint64_t size() const noexcept {
                return high_water_;
            }

        protected:
            std::streamsize xsputn(
                const char*,
                const std::streamsize count) override {
                if (count < 0) {
                    return 0;
                }
                advance(static_cast<std::uint64_t>(count));
                return count;
            }

            int_type overflow(const int_type character) override {
                if (traits_type::eq_int_type(
                        character, traits_type::eof())) {
                    return traits_type::not_eof(character);
                }
                advance(1);
                return character;
            }

            pos_type seekoff(
                const off_type offset,
                const std::ios_base::seekdir direction,
                const std::ios_base::openmode mode) override {
                if ((mode & std::ios_base::out) == 0) {
                    return pos_type(off_type(-1));
                }
                off_type base = 0;
                if (direction == std::ios_base::beg) {
                    base = 0;
                } else if (direction == std::ios_base::cur) {
                    base = static_cast<off_type>(cursor_);
                } else if (direction == std::ios_base::end) {
                    base = static_cast<off_type>(high_water_);
                }
                if (offset < -base || offset > std::numeric_limits<off_type>::max() - base) {
                    return pos_type(off_type(-1));
                }
                const auto next =
                    static_cast<std::uint64_t>(base + offset);
                cursor_ = next;
                high_water_ =
                    std::max(high_water_, cursor_);
                return pos_type(
                    static_cast<off_type>(cursor_));
            }

            pos_type seekpos(
                const pos_type position,
                const std::ios_base::openmode mode) override {
                return seekoff(
                    static_cast<off_type>(position),
                    std::ios_base::beg, mode);
            }

        private:
            void advance(const std::uint64_t bytes) {
                if (bytes >
                    static_cast<std::uint64_t>(std::numeric_limits<off_type>::max()) -
                        cursor_) {
                    throw std::overflow_error(
                        "Checkpoint byte count overflows");
                }
                cursor_ += bytes;
                high_water_ =
                    std::max(high_water_, cursor_);
            }

            std::uint64_t cursor_ = 0;
            std::uint64_t high_water_ = 0;
        };

}

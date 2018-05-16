#ifndef RKTK_FILENAME_HELPERS_HPP_INCLUDED
#define RKTK_FILENAME_HELPERS_HPP_INCLUDED

#include <cstddef> // for std::size_t
#include <cstdint> // for std::uint64_t
#include <ios> // for std::dec and std::hex
#include <sstream> // for std::istringstream
#include <string>

static inline bool is_dec_digit(char c) {
    return ('0' <= c) && (c <= '9');
}

static inline bool is_hex_digit(char c) {
    return (('0' <= c) && (c <= '9'))
           || (('a' <= c) && (c <= 'f'))
           || (('A' <= c) && (c <= 'F'));
}

static inline bool is_dec_substr(const std::string &str,
                                 std::string::size_type begin,
                                 std::string::size_type end) {
    for (std::string::size_type i = begin; i < end; ++i) {
        if (!is_dec_digit(str[i])) { return false; }
    }
    return true;
}

static inline bool is_hex_substr(const std::string &str,
                                 std::string::size_type begin,
                                 std::string::size_type end) {
    for (std::string::size_type i = begin; i < end; ++i) {
        if (!is_hex_digit(str[i])) { return false; }
    }
    return true;
}

static inline bool is_rktk_filename(const std::string &filename) {
    if (filename.size() != 68) { return false; }
    if (!is_dec_substr(filename, 0, 4)) { return false; }
    if (filename[4] != '-') { return false; }
    if (!is_dec_substr(filename, 5, 9)) { return false; }
    if (filename[9] != '-') { return false; }
    if (filename[10] != 'R') { return false; }
    if (filename[11] != 'K') { return false; }
    if (filename[12] != 'T') { return false; }
    if (filename[13] != 'K') { return false; }
    if (filename[14] != '-') { return false; }
    if (!is_hex_substr(filename, 15, 23)) { return false; }
    if (filename[23] != '-') { return false; }
    if (!is_hex_substr(filename, 24, 28)) { return false; }
    if (filename[28] != '-') { return false; }
    if (!is_hex_substr(filename, 29, 33)) { return false; }
    if (filename[33] != '-') { return false; }
    if (!is_hex_substr(filename, 34, 38)) { return false; }
    if (filename[38] != '-') { return false; }
    if (!is_hex_substr(filename, 39, 51)) { return false; }
    if (filename[51] != '-') { return false; }
    if (!is_dec_substr(filename, 52, 64)) { return false; }
    if (filename[64] != '.') { return false; }
    if (filename[65] != 't') { return false; }
    if (filename[66] != 'x') { return false; }
    return (filename[67] == 't');
}

static inline std::size_t dec_substr_to_int(const std::string &str,
                                            std::string::size_type begin,
                                            std::string::size_type end) {
    std::istringstream substr_stream(str.substr(begin, end - begin));
    std::size_t result;
    substr_stream >> std::dec >> result;
    return result;
}

static inline std::uint64_t hex_substr_to_int(const std::string &str,
                                              std::string::size_type begin,
                                              std::string::size_type end) {
    std::istringstream substr_stream(str.substr(begin, end - begin));
    std::uint64_t result;
    substr_stream >> std::hex >> result;
    return result;
}

#endif // RKTK_FILENAME_HELPERS_HPP_INCLUDED

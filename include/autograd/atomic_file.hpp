#pragma once

#include <filesystem>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace autograd {
namespace atomic_file_detail {

inline std::filesystem::path temporary_path(const std::filesystem::path& destination) {
    if (destination.empty() || destination.filename().empty())
        throw std::invalid_argument("Atomic-save destination must name a file");
    std::filesystem::path temporary = destination;
    temporary += ".tmp";
    return temporary;
}

inline void replace_file(const std::filesystem::path& temporary,
                         const std::filesystem::path& destination) {
    std::filesystem::rename(temporary, destination);
}

template <class Writer, class Replacer>
void write_file(const std::filesystem::path& destination,
                Writer&& writer,
                Replacer&& replacer) {
    const std::filesystem::path temporary = temporary_path(destination);
    std::error_code status_error;
    const bool temporary_exists = std::filesystem::exists(temporary, status_error);
    if (status_error)
        throw std::filesystem::filesystem_error(
            "Unable to inspect atomic-save temporary path", temporary, status_error);
    if (temporary_exists)
        throw std::runtime_error("Atomic-save temporary path already exists: " +
                                 temporary.string());

    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Unable to create atomic-save temporary file: " +
                                 temporary.string());

    try {
        std::forward<Writer>(writer)(output);
        output.flush();
        if (!output) throw std::runtime_error("Failed to flush atomic-save temporary file");
        output.close();
        if (!output) throw std::runtime_error("Failed to close atomic-save temporary file");
        std::forward<Replacer>(replacer)(temporary, destination);
    } catch (...) {
        output.close();
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw;
    }
}

} // namespace atomic_file_detail

template <class Writer>
void atomic_write_file(const std::filesystem::path& destination, Writer&& writer) {
    atomic_file_detail::write_file(destination, std::forward<Writer>(writer),
                                   atomic_file_detail::replace_file);
}

} // namespace autograd

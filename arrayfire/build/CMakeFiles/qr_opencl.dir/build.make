# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alameddin/test/arrayfire

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alameddin/test/arrayfire/build

# Include any dependencies generated for this target.
include CMakeFiles/qr_opencl.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/qr_opencl.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/qr_opencl.dir/flags.make

CMakeFiles/qr_opencl.dir/qr.cpp.o: CMakeFiles/qr_opencl.dir/flags.make
CMakeFiles/qr_opencl.dir/qr.cpp.o: ../qr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alameddin/test/arrayfire/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/qr_opencl.dir/qr.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/qr_opencl.dir/qr.cpp.o -c /home/alameddin/test/arrayfire/qr.cpp

CMakeFiles/qr_opencl.dir/qr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/qr_opencl.dir/qr.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alameddin/test/arrayfire/qr.cpp > CMakeFiles/qr_opencl.dir/qr.cpp.i

CMakeFiles/qr_opencl.dir/qr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/qr_opencl.dir/qr.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alameddin/test/arrayfire/qr.cpp -o CMakeFiles/qr_opencl.dir/qr.cpp.s

# Object files for target qr_opencl
qr_opencl_OBJECTS = \
"CMakeFiles/qr_opencl.dir/qr.cpp.o"

# External object files for target qr_opencl
qr_opencl_EXTERNAL_OBJECTS =

qr_opencl: CMakeFiles/qr_opencl.dir/qr.cpp.o
qr_opencl: CMakeFiles/qr_opencl.dir/build.make
qr_opencl: /usr/local/lib64/libafopencl.so.3.7.0
qr_opencl: CMakeFiles/qr_opencl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alameddin/test/arrayfire/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable qr_opencl"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/qr_opencl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/qr_opencl.dir/build: qr_opencl

.PHONY : CMakeFiles/qr_opencl.dir/build

CMakeFiles/qr_opencl.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/qr_opencl.dir/cmake_clean.cmake
.PHONY : CMakeFiles/qr_opencl.dir/clean

CMakeFiles/qr_opencl.dir/depend:
	cd /home/alameddin/test/arrayfire/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alameddin/test/arrayfire /home/alameddin/test/arrayfire /home/alameddin/test/arrayfire/build /home/alameddin/test/arrayfire/build /home/alameddin/test/arrayfire/build/CMakeFiles/qr_opencl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/qr_opencl.dir/depend


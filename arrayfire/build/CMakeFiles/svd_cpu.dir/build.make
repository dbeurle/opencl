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
include CMakeFiles/svd_cpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/svd_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/svd_cpu.dir/flags.make

CMakeFiles/svd_cpu.dir/svd.cpp.o: CMakeFiles/svd_cpu.dir/flags.make
CMakeFiles/svd_cpu.dir/svd.cpp.o: ../svd.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alameddin/test/arrayfire/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/svd_cpu.dir/svd.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svd_cpu.dir/svd.cpp.o -c /home/alameddin/test/arrayfire/svd.cpp

CMakeFiles/svd_cpu.dir/svd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svd_cpu.dir/svd.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alameddin/test/arrayfire/svd.cpp > CMakeFiles/svd_cpu.dir/svd.cpp.i

CMakeFiles/svd_cpu.dir/svd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svd_cpu.dir/svd.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alameddin/test/arrayfire/svd.cpp -o CMakeFiles/svd_cpu.dir/svd.cpp.s

# Object files for target svd_cpu
svd_cpu_OBJECTS = \
"CMakeFiles/svd_cpu.dir/svd.cpp.o"

# External object files for target svd_cpu
svd_cpu_EXTERNAL_OBJECTS =

svd_cpu: CMakeFiles/svd_cpu.dir/svd.cpp.o
svd_cpu: CMakeFiles/svd_cpu.dir/build.make
svd_cpu: /usr/local/lib64/libafcpu.so.3.7.0
svd_cpu: CMakeFiles/svd_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alameddin/test/arrayfire/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable svd_cpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/svd_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/svd_cpu.dir/build: svd_cpu

.PHONY : CMakeFiles/svd_cpu.dir/build

CMakeFiles/svd_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/svd_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/svd_cpu.dir/clean

CMakeFiles/svd_cpu.dir/depend:
	cd /home/alameddin/test/arrayfire/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alameddin/test/arrayfire /home/alameddin/test/arrayfire /home/alameddin/test/arrayfire/build /home/alameddin/test/arrayfire/build /home/alameddin/test/arrayfire/build/CMakeFiles/svd_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/svd_cpu.dir/depend


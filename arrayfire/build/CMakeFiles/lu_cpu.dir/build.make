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
include CMakeFiles/lu_cpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lu_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lu_cpu.dir/flags.make

CMakeFiles/lu_cpu.dir/lu.cpp.o: CMakeFiles/lu_cpu.dir/flags.make
CMakeFiles/lu_cpu.dir/lu.cpp.o: ../lu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alameddin/test/arrayfire/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lu_cpu.dir/lu.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lu_cpu.dir/lu.cpp.o -c /home/alameddin/test/arrayfire/lu.cpp

CMakeFiles/lu_cpu.dir/lu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lu_cpu.dir/lu.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alameddin/test/arrayfire/lu.cpp > CMakeFiles/lu_cpu.dir/lu.cpp.i

CMakeFiles/lu_cpu.dir/lu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lu_cpu.dir/lu.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alameddin/test/arrayfire/lu.cpp -o CMakeFiles/lu_cpu.dir/lu.cpp.s

# Object files for target lu_cpu
lu_cpu_OBJECTS = \
"CMakeFiles/lu_cpu.dir/lu.cpp.o"

# External object files for target lu_cpu
lu_cpu_EXTERNAL_OBJECTS =

lu_cpu: CMakeFiles/lu_cpu.dir/lu.cpp.o
lu_cpu: CMakeFiles/lu_cpu.dir/build.make
lu_cpu: /usr/local/lib64/libafcpu.so.3.7.0
lu_cpu: CMakeFiles/lu_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alameddin/test/arrayfire/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lu_cpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lu_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lu_cpu.dir/build: lu_cpu

.PHONY : CMakeFiles/lu_cpu.dir/build

CMakeFiles/lu_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lu_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lu_cpu.dir/clean

CMakeFiles/lu_cpu.dir/depend:
	cd /home/alameddin/test/arrayfire/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alameddin/test/arrayfire /home/alameddin/test/arrayfire /home/alameddin/test/arrayfire/build /home/alameddin/test/arrayfire/build /home/alameddin/test/arrayfire/build/CMakeFiles/lu_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lu_cpu.dir/depend


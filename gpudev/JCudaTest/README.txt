This is a test application that uses jCuda (http://www.jcuda.org/)

Setup of dependencies:

1. Download jCuda's sources, manually change architecture in CMakeFiles.txt for ppc64le
and compile. Now you have the native dynamic libraries in the lib folder.
If you don't have optional libs like BLAS, just comment them out. Needed ones are
CUDA common, runtime and driver.

2. Download mavenized-jcuda:
git clone https://github.com/MysterionRise/mavenized-jcuda
Then add a new profile, just like unix-x86_64 one, but with <arch>ppc64le</arch>
and changed names. Then put into repo/jcuda/libJCudaRuntime/0.7.0a/ the .so file from
the previous step with name changed to libJCudaRuntime-0.7.0a-linux-ppc_64.so.
Do the same for the driver file. Then compile mavenized-cuda with
mvn clean install
Check in log that *.so files are copied properly. Then run
mvn exec:exec
If everything is fine, a program displaying native pointer address with jCuda should run.
This part is already done and is in /home/janw/jcuda/mavenized-jcuda.

Setup of a jCuda (already done):

1. mavenized-jcuda has to be added to dependencies.
2. maven-dependency-plugin has to be used with configuration option
<stripVersion>true</stripVersion> or else the library files will be copied with version
and JNI (Java Native Interface) won't be able to find them.
3. exec-maven-plugin has to be added to dependencies for the purpose of running everything
automatically with correct classpath and library path. To run the stuff with scala,
the arguments should be like that:
<configuration>
    <executable>scala</executable>
    <arguments>
        <argument>-Djava.library.path=${project.build.directory}/lib</argument>
        <argument>-classpath</argument>
        <classpath/>
        <argument>NameOfMainClass</argument>
    </arguments>
</configuration>

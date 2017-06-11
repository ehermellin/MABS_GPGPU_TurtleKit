**A Method for Adapting MABS Models to GPU Programming**
-------------
Multi-Agent Based Simulation (MABS) may require to handle a large number of agents. As this number is constantly growing, the computational resources which are needed are hardly fulfilled by the Central Processing Unit (CPU). Considering this issue, General-Purpose Computing on Graphics Units (GPGPU) appears to be very promising as it allows to use the massively parallel architecture of the graphics cards to do general-purpose computing. However, this technology relies on a highly specialized architecture, implying a very specific programming approach. So, a MAS model cannot benefit from GPU power without having been adapted to the GPU programming paradigm. To ease the use of this technology, most of the recent research works propose solutions that hide GPU programming, making its use completely transparent. However, because of the wide heterogeneity of multi-agent models, such solutions only focus on some particular type of MAS models and are not suited for all cases. So, this paper is motivated by the idea that the MABS community would benefit from a real methodology that would (1) help potential users to decide if they could benefit from GPGPU considering their models and (2) support the translation of MABS model into GPU programming without hiding the underlying technology. In this paper, we present a methodology dedicated to GPGPU in a MABS context which defines the iterative process to be followed to transform a model so that it takes advantage of the GPU power. To this end, we use the GPU environmental delegation principle (based on an hybrid approach) on several models and analyze the result of this experiment to identify recurrent tasks and thus introduce the methodology.

> **Note:** 

> https://dl.acm.org/citation.cfm?id=2936924.2937106

> http://dx.doi.org/10.1007/978-3-319-39387-2_11

----------

### About GPU programming
To program on the graphics card and exploit its GPGPU capabilities, we use CUDA which is a programming interface dedicated to GPGPU provided by Nvidia. The programming model relies on the following philosophy: The CPU is called the host and plays the role of scheduler. The host manages data and triggers kernels, which are functions specifically designed to be executed by the GPU, which is called the device. The GPU part of the code really differs from sequential code and has to fit the underlying hardware architecture. More precisely, the GPU device is programmed to proceed the parallel execution of the same procedure, the kernel, by means of numerous threads. These threads are organized in blocks (the parameters blockDim.x, blockDim.y characterize the size of these blocks), which are themselves structured in a global grid of blocks. Each thread has unique 3D coordinates (threadIdx.x, threadIdx.y, threadIdx.z) that specifies its location within a block. Similarly, each block also has three spatial coordinates (respectively blockIdx.x, blockIdx.y, blockIdx.z) that localize it in the global grid. The figure below illustrates this organization for the 2D case. So each thread works with the same kernel but uses different data according to its spatial location within the grid. Moreover, each block has a limited thread capacity according to the hardware in use.

> **Note:** Thread is similar to the concept of task. Indeed, a thread may be considered as an instance of the kernel which is performed on a restricted portion of the data depending on its location in the global grid (its identifier).

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/grilleblocthreadgpgpu.png)

----------

### About the method
The application of the GPU delegation principle on a model can be divided into four main phases. The first step consists in identifying in the model all the agents' behavior and the various computations and dynamics present in the environment. Once all this information referenced, it is necessary to browse them to identify which one is eligible to a delegation into environmental dynamics and computes by a GPU module. This selection of computations corresponds to the second step. The third step consists in analyzing the selected computations and then choosing which would bring the most interesting gains once translated in GPU modules. Finally, step 4 consists in applying the principle on the identified computations.
So, the application pattern of the GPU delegation principle can be summarized as follows:
![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/Methodologygraph.png)

----------

### Experimenting the Methodology on 4 Models

> **Note:** To test our implementation follow the link : https://github.com/fmichel/TurtleKit

#### Game of Life

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/gameoflife.jpg)

The Conway's Game of Life model does not contain agents (it's a cellular automaton) but it has environmental dynamics representative of those that one encounters in MAS (and it is often used in the community). The environment of the Game of Life is an infinite two dimensional orthogonal cellular automaton with a two dimensional grid of square cells, each of which is in one of two possible states: Dead or alive. Every cell interacts with its eight neighbors according to the following rules: (1) Any live cell with fewer than two and more than three live neighbors dies, (2) any live cell with two or three live neighbors lives and (3) any dead cell with exactly three live neighbors becomes a live cell. In our experiments, the grid is initialized with a random cell pattern (the probability for the cell to be dead or alive is the same).

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/Gameoflifegraph.png)

The main computational part of the model is located in step (1) which consists in computing a sequential loop: It calculates the new state of each cell for the next step of the simulation. So, the more the environment is large, the more this computation is long. It may therefore be advantageous to transform this computation into a GPU module.
To create the GPU module correctly, it is necessary to focus on the representation of data. Especially, to avoid expensive transfers between CPU and GPU, the data need to be sent only once at each step. To this end, each cell write its state value (1 for alive and 0 for dead) in a 2D array (matching the size of the environment) according to its position. Then, this array is sent to the GPU. This module does the sum of the states (more precisely the sum of the number of alive cells) of all Moore neighborhood cells for each cell of the grid and stores the result in the 2D result array. Thus, each cell of the result array contains a value between 0 (no cell alive) and 8 (all neighbors are alive). The result array is then used to update the grid and compute the next simulation step according to the transition rules.

``` C
Input:   width, height, statesArray
Output:  resultArray (the number of alive cells)

i = blockIdx.x * blockDim.x + threadIdx.x;
j = blockIdx.y * blockDim.y + threadIdx.y;
sumOfState = 0;

if(i < width and j < height){
  sumOfState = getNeighborsState(statesArray[i,j]);
}
resultArray[i,j] = sumOfState;
```

#### Schelling Segregation

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/segregation.jpg)

The Schelling's Segregation model models the behavior of two types of agents in a neighborhood: Red agents and green agents. These agents are scattered across a two dimensional grid. Their purpose is to be happy by staying near like-colored agents. If they are dissatisfied at their position, agents attempt to move to a new random vacant location in order to find a better place according to their objective of happiness. In our experiments, green and red agents are scattered with the same proportion. The initial distribution in the environment is made in a random way.

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/Segregationgraph.png)

The most intensive computations are in step (2) and (3) which consist, for each agent, in recovering a neighbor list and then computing its happiness according to the list. So, many sequential loops have to be done at each time step, and the required computation time clearly increases depending on the number of agents. According to the GPU delegation, the happiness computation can be deported into an environmental dynamics because these computations do not modify agents' states.
In the CPU model, the computation of happiness is done by testing the color of the agents present in the list of neighbors and counting the number of agents in each community according to their state: 1 for green agents and -1 for red agents. However, if we want to deport this computation, the data structure sent to the GPU needs to be adapted. So, at each time step, agents write in a 2D array (matching the size of the environment) their states depending on their position. This table is sent to the GPU that computes the sum of the neighboring states and returns in a result array the values for each cell of the environment. This result array thus contains values between -8 (all agents around are red) and 8 (all agents around are green). The agents then recover the value in the result array with respect to their position and act accordingly.

``` C
Input:   width, height, communityArray
Output:  resultArray (happiness quantity)

i = blockIdx.x * blockDim.x + threadIdx.x;
j = blockIdx.y * blockDim.y + threadIdx.y;
sumOfCommunityState = 0;

if(i < width and j < height){
  for(agent in getNeighborsCommunity()){
    agentCommunityState = communityArray[i,j];
    if(agentCommunityState == 1){
      sumOfCommunityState ++;
    }
    if(agentCommunityState == -1){
      sumOfCommunityState --$;
    }
  }
}
resultArray[i,j] = sumOfCommunityState;
```

#### Schelling Segregation

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/fire.jpg)

The Fire model simulates the spread of a fire through a forest. In this model, all the trees are agents placed randomly in the environment (a two dimensional grid). Trees can be alive, burned or died. When it burns, a tree releases heat which spreads in the environment. This heat can ignite other trees around: A tree ignites when the temperature is above a defined threshold. The threshold of each agent is randomly set within a range of values. A tree dies when its life reaches zero: The life of the tree decreases when it burns.

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/Firegraph.png)

The most greedy computation loop of this model is in step (1). Indeed, the environment must compute the heat diffusion which requires making a global sequential loop over all the cells. Thus, the computation time increases very quickly according to the size of the environment. Because the heat diffusion is already an environmental dynamics, the delegation of this computation is very easy because it can be directly translated into a GPU module.
So, in the GPU model, the heat diffusion is done as follows: At each time step, agents add in a 2D array (matching the size of the environment) the heat that they release according to their state (alive, burn or dead). Then, this array is sent to the GPU module that compute the sum of heat values from neighboring cells (Moore neighborhood here) for each cell of the environment. More preciselly, the sum consists in adding the heat values already present in the environment (from the previous steps) and the heat generated by the agents, all modulated by a diffusion variable. Once the computation performed, the agents recover the heat value in the array with respect to their position and act accordingly.

``` C
Input:   width$, $height$, $heatArray$, $radius$}
Output:  resultArray$ (the quantity of heat)}

i = blockIdx.x * blockDim.x + threadIdx.x;
j = blockIdx.y * blockDim.y + threadIdx.y;
sumOfHeat = 0;

if(i < width and j < height){
  sumOfHeat = getNeighborsHeat(heatArray[i,j], radius);
}
resultArray[i,j] = sumOfHeat * heatAdjustment;
```

#### Schelling Segregation

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/dla.jpg)

The DLA model demonstrates diffusion-limited aggregation, in which randomly moving particles stick together to form treelike branching fractal structures. In this model, particles are agents who are initialized and move in a random way into the environment (a two dimensional grid). Randomly, one of the agents stops and stays at the same position. Then, when a moving red agent encounters a motionless green agent, it stops and turns into a green agent (the other agents continue to move in a random way).

![Threads Blocs Grid](https://github.com/ehermellin/MABS_GPGPU_TurtleKit/raw/master/images/Dlagraph.png)

The most greedy computations are in step (2) and (3) which consist for each agent in recovering a neighbor list and then searching within this list the nearest neighbors. So, the computation time of this model greatly increases according to the number of agents. According to GPU delegation, the agent action which consists in checking if one of the neighbor agents is green (motionless) can be converted into an environmental dynamics and then performed by a GPU kernel because this computation do not modify agents' states.
However, it is necessary to adapt the data structure to enable this transformation. Thus, all agents report, according to their position, their presence in a 2D array (matching the size of the environment): 1 if an agent occupies the cell, 0 otherwise. This array is sent to the GPU that calculates for each cell if there are agents around it. All the cells of the result array thus contain a value representing the number of neighbor agents (0 for an empty cell, 1 or more means there are neighbors around). So, the agents only have to recover the result value in the array and adjust their behavior accordingly. Algorithm \ref{algo:dla} presents an implementation of the corresponding GPU kernel.

``` C
Input:   width$, $height$, $presenceArray$}
Output:  resultArray$ (the number of neighbors)}

i = blockIdx.x * blockDim.x + threadIdx.x;
j = blockIdx.y * blockDim.y + threadIdx.y;
sumOfAgents = 0;

if(i < width and j < height){
  	sumOfAgents = getNeighborsPresence(presenceArray[i,j])\;
}
resultArray[i,j] = sumOfAgents;
```

----------
# DADS
 💻 欢迎在云边协同领域工作的同学一起交流

 💻 如果有一些代码中的bug，请提出issue，我尽量完善

🥳 本项目根据经典论文进行复现：Dynamic adaptive DNN surgery for inference acceleration on the edge

论文链接🔗：https://github.com/Tjyy-1223/Neurosurgeon/blob/main/paper/Collaborative_Intelligence%20Between_the_Cloud_and_Mobile_Edge.pdf

## 研究内容

**DNN的最新进展表明DNN不再局限于链式拓扑，DAG拓扑变得流行。**

例如，GoogleNet和ResNet是DAG结构。显然，将DAG图划分代替链式划分涉及更复杂的图论问题。设计了一种动态自适应DNN手术(DADS)方案，通过持续监测DNN网络状态来优化分割DNN网络。

DADS的关键设计如下：

- DADS持续监测网络状况，并确定系统是在轻载状态还是重负载状态下运行。
- **轻载条件下，**设计DSL算法，以最小化处理一帧的总延迟。在这一部分中，为了求解时延最小化问题，我们将原问题转化为一个等价的最小割问题，以便找到全局最优解。
- **高负载情况下，**提出了DSH算法，最大化吞吐量，即单位时间内可以处理的帧数。然而，本文证明了这类优化问题是np -难问题，不能在多项式时间复杂度内求解。DSH采用近似方法，其近似比为3。

![image-20230709090527770](./assets/image-20230709090527770.png)

## 项目结构

```

```

## 运行环境

```
python 3.9
torch==1.9.0.post2
torchvision==0.10.0
xlrd==2.0.1
apscheduler
```

## 项目运行

补充完善中...

## 论文细节

#### A. The Impact of DNN Inference Workloads

<img src="/Users/tianjiangyu/MyStudy/云边协同研究/DADS/assets/image-20230709092724650.png" alt="image-20230709092724650" style="zoom: 67%;" />

**（1）在轻工作负载下是这样的:对于每个阶段，当前帧在下一帧到达之前完成。如图8下面那个图展示**

数学上表示为
$$
max\{t_c,t_t,t_e \} \lt 1/Q
$$
甘特图表示如图8，这种情况下，我们只需要尽快完成每一帧，即最小化 
$$
T_e + T_c + T_t
$$
**（2）当系统负载较大时，最小化Te+Tt+ Tc可能会导致系统拥塞**
$$
max\{t_c,t_t,t_e \} \ge 1/Q
$$
例如，在图8中靠上的子图，由于边缘设备上推理时延过长，使下一帧在当前帧完成边缘之前到达。因此，在这种情况下，**我们需要最大化系统的吞吐量**，即最小化 
$$
max\{t_c,t_t,t_e \}
$$
即最大化系统吞吐量：在单位时间内可以处理的任务数量。



#### B. The Light Workload Partitioning Algorithm

<img src="./assets/image-20230709094554166.png" alt="image-20230709094554166" style="zoom: 67%;" />

**构建了一个新的图$G'$，使每个边只捕获一个延迟值,将ECDI-L问题转化为G的最小加权s-t切割问题。**

首先介绍了如何在G的基础上构造G。

- Cloud Computing Delay：red links，用来捕获云计算时延
- Edge Computing Delay：blue links，以捕获v的边缘计算延迟。
- Communication Delay：其他所有links对应通信时延，为解决如图中v1有多个后继节点，所以引入了v1'。

红、蓝、黑对应成本是云计算、边缘计算和通信延迟。虚线链接被赋予无穷大。

**ECDI-L等价于G的最小e-c cut。**

- 如果对e到vi的链接进行切割(图9(b)所示的红色链接)，则vi将在云端进行处理
- 如果从vj到c进行链接切割(图9(b)中的蓝色链接)，则将vj加工在边缘上
- 如果对vi到vj进行链路切割(图9(b)所示黑色链路)，则将vi的数据传输到云端

红色链路的总切割成本等于云计算时间Tc。蓝色边的总切割成本等于边的计算时间Te。黑色链路的总剪切代价等于不考虑网络延迟Tt的传输时间。如果G的e-c cut最小，则单个帧上的推理延迟最小。

![image-20230709095005389](/Users/tianjiangyu/MyStudy/云边协同研究/DADS/assets/image-20230709095005389.png)



#### C. The Heavy Workload Partitioning Algorithms

![image-20230709095546622](./assets/image-20230709095546622.png)



#### D. The Dynamic Partitioning Algorithm

<img src="./assets/image-20230709091256841.png" alt="image-20230709091256841" style="zoom:67%;" />

网络状态是变化的，将动态地影响工作负载模式的选择和分区决策；设计了DADS以适应网络的动态变化。如算法3所示：

- monitor-task()监控视频是否活动(第2行)。这可以通过“iperf”工具实现。具体实现参见第四节。
- 实时网络带宽由monitor-net()(第3行)推导得到
- 调用DSL()计算分区策略(第4行)。
- 此时，如果处于轻负载模式，并且通过DSL进行分区是被接受的；否则，系统处于重负载模式，调用DSH()调整分区策略以最小化最大延迟(第6行)。然而，如果完成率仍然小于采样率，则说明采样率过大，即使DSH()也不能满足采样率。系统将会拥塞。它调用用户降低采样率(第7-8行)。

## 交流

如果对本项目有更好的想法或者交流，可以在GitHub Issue提出问题

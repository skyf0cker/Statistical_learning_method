# ESRGAN

## abstract：

1. SRGAN 效果不好，具体在于生成幻觉效果，视觉效果不好
2. 引入RRDB参差密集块，不进行规范化

## Introduction

1. SRCNN好，先驱，但是PSNR这个标准就有问题，和人感觉不一致，得到结果过度平滑

2. SRGAN好，将GAN引入SR问题，视觉效果提升，基本架构：残差块，但是仍有差距

3. 他们牛逼，从三方面改进

   - 引入剩余残差密集块（RDDB）来改进网络结构，删除了(BN)批量标准化，使用残差缩放和更小的初始化来促进训练非常深的网络
   - 使用RaGAN改进鉴别器，它可以判断“一个图像是否比另一个更真实”，而不是“一个图像是真实的还是假的”。
   - 提出通过在激活之前使用VGG特征来改善感知损失，而不是像SRGAN那样激活之后。

4. PIRM采用了无参考的评测方法，就是PI值

   根据RMSE的值，结果分三个区，每个区的最低分为地区冠军

5. 为了平衡视觉质量和RMSE / PSNR，提出了网络插值策略，可以不断调整重构风格和平滑度。另一种替代方案是图像插值，其逐个像素地直接插值图像。

##　related work

1. 出现许多模型：

   - a deeper network with residual learning

   - Laplacian pyramid structure
   -  residual blocks
   - recursive learning
   - densely connected net-work 
   - deep back projection 
   - residual dense network

2. 真正有用的：

   - 通过去除残余块中不必要的BN层并扩展模型尺寸来提出EDSR模型，从而实现了显着的改进
   - 建议在SR中使用有效残余密集块，并进一步探索具有信道关注的更深层网络[12]，从而实现最先进的PSNR性能

3. 出现许多模型来训练贼深的网络：

   - 开发 residual path以稳定训练并改善性能
   - 采用残余缩放 Residual scalin
   - 没有BN的VGG型网络

4. WGAN 提出通过权重削减来最小化Wasserstein距离和正则化判别器的合理和有效近似，

   于是他们采用了RaGan

## proposed Methods

1. 网络模型：

   ![1550232222938](/home/vophan/.config/Typora/typora-user-images/1550232222938.png)

   采用SRResNet的基本架构，其中大多数计算都是在LR特征空间中完成的。可以选择或设计“基本块”（例如，残余块，密集块，RRDB）以获得更好的性能

   - 我们主要对发生器G的结构进行两处修改：

     1）去除所有BN层;

     2）用建议的剩余残差密码块（RRDB）替换原始基本块，RRRP结合了多级残留网络和密集连接

     ![1550232384120](/home/vophan/.config/Typora/typora-user-images/1550232384120.png)

     当训练和测试数据集的统计数据大量增加时，BN层往往会引入令人不快的伪影并限制一般化能力，而且去掉BN层，还会减少计算量。

     保留了SRGAN的高级架构设计，并使用了如图4所示的基本块（即RRDB）。基于观察，**更多的层和连接总能提升性能，提出的RRDB采用比SRGAN中原始残差块更深，更复杂的结构**。使用的RRDB与其他的不同之处在于主路径中使用denseblock ，其中网络容量变得更高，受益于密集连接

     残差缩放，**即通过在0到1之间乘以常数来缩减残差**，然后将它们添加到主路径以防止不稳定。

     较小的初始化，因为当初始参数方差变小时，我们经验地发现残差结构更容易训练。

2. 相对的判别模型

   与标准鉴别器Din SRGAN不同，**它估计一个输入图像是真实和自然的概率，**相对论鉴别器试图预测realimagexris相对比假onexf更逼真的概率.

   ![1550233657695](/home/vophan/.config/Typora/typora-user-images/1550233657695.png)

3. 他们还改进了Loss function

4. 网络差值

![1550236666392](/home/vophan/.config/Typora/typora-user-images/1550236666392.png)


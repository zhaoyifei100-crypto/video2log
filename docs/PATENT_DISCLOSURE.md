# 专利技术交底书

## 发明名称

**基于背照式图像传感器面对面键合与阻变存储器的低功耗边缘视觉感知芯片架构**

---

## 技术领域

本发明涉及半导体集成电路设计领域，具体涉及一种结合三维堆叠架构与阻变存储器（RRAM）技术的边缘视觉感知系统。特别适用于车载视觉系统、机器人视觉、智能监控等需要超低功耗、超低延迟及可配置智能感知的边缘计算场景。

---

## 背景技术

### 现有技术现状

当前车载视觉系统和机器人视觉系统普遍采用以下架构：

1. **传统串行传输架构**
   ```
   摄像头模组 → MIPI/GMSL串行输出 → 解串器 → 主SOC → 内存 → AI加速器
   ```
   
   该架构存在以下问题：
   - **高功耗**：串行编解码（SerDes）功耗占系统总功耗的30-50%
   - **高延迟**：从图像采集到处理完成延迟通常在50-100ms
   - **带宽瓶颈**：多路摄像头同时工作时，总线带宽成为系统瓶颈
   - **无效数据传输**：大部分时间传输的是无事件的背景画面

2. **近传感器计算架构（Near-Sensor Computing）**
   
   现有技术尝试将部分计算移到传感器附近，但仍存在：
   - 传感器与处理芯片通过PCB走线连接，距离远、干扰大
   - 仍需标准化的串行接口（MIPI CSI-2），无法充分利用像素级并行性
   - 需要大容量SRAM或外部存储器存储背景模型，功耗和面积开销大
   - 传统背景建模算法（帧差法、GMM）参数固定，适应性差

3. **现有BSI图像传感器架构**
   
   当前BSI（Back-Side Illumination）CMOS图像传感器采用**单芯片（monolithic）设计**：
   - 光电二极管阵列位于硅片背部（接收光子）
   - 晶体管逻辑电路（ADC、控制逻辑）位于硅片前部
   - 两者通过金属互连层在同一die内连接
   
   **局限性**：
   - 传感器通过标准封装（BGA/LGA）引出到PCB
   - 需经过PCB走线连接外部处理芯片
   - 再通过串行接口（MIPI CSI-2/GMSL）传输数据
   - 无法充分利用像素级并行性，功耗和延迟受限

### 现有技术的缺陷总结

1. **功耗问题**：传统架构在传输无事件图像时浪费大量功耗；事件检测需要大容量存储器
2. **延迟问题**：串行传输和协议转换引入不可接受的延迟
3. **存储瓶颈**：背景模型存储需要SRAM/PSRAM，功耗高、面积大
4. **适应性差**：传统CV算法参数固定，无法针对不同场景优化
5. **集成度**：多芯片方案占用PCB面积大，成本高

---

## 发明内容

### 发明目的

本发明提供一种基于背照式（BSI）图像传感器面对面键合与阻变存储器（RRAM）的芯片架构，实现：
- **超低功耗**：仅在检测到有效事件时激活高速数据传输，事件检测功耗<10mW
- **超低延迟**：微秒级的感知-响应延迟
- **存储优化**：使用RRAM存储神经网络权重，无需大容量SRAM/PSRAM
- **可配置性**：出厂前可对权重进行微调，适应不同应用场景
- **高集成度**：单封装内完成光电转换到智能决策的全流程
- **即插即用**：兼容现有GMSL/FPD-Link车载协议，可直接替换传统摄像头模组

### 核心创新点

1. **Face-to-Face键合架构**
   
   现有BSI图像传感器采用单芯片设计，光电二极管和逻辑电路集成在同一die上，通过PCB走线连接外部处理芯片。本发明提出一种新型的**双芯片Face-to-Face堆叠架构**：
   - 第一芯片：BSI图像传感器（如OmniVision的独立die），光线从背面入射，逻辑电路位于正面
   - 第二芯片：边缘感知处理芯片
   - 两芯片通过**混合键合（Hybrid Bonding）**或微凸点键合实现面对面直接连接
   - 省去PCB走线和传统封装引脚，实现像素级并行数据传输

2. **像素级并行接口**
   
   通过面对面键合实现数千条并行数据线，直接传输像素级数据，跳过串行编码阶段。

3. **基于RRAM的神经网络事件检测**
   
   采用创新的技术方案解决传统事件检测的存储瓶颈：
   - 使用**阻变存储器（RRAM）**存储训练好的卷积神经网络权重
   - 权重在出厂前通过有限次写入（fine-tune）进行配置
   - 推理阶段RRAM为只读模式，功耗极低
   - 无需大容量SRAM存储背景模型，显著降低功耗和面积

4. **数字MAC阵列加速器**
   
   - 采用全数字计算架构（Option A），RRAM仅用于存储，计算使用传统数字MAC单元
   - 支持INT8/INT4量化推理
   - 峰值算力：1-10 GOPS，功耗<10mW

5. **事件触发的高速传输机制**
   
   处理芯片内部集成基于神经网络的事件检测引擎，仅当检测到有效事件时才激活高速接口（HSMT/GMSL）向主SOC传输数据，平时保持idle状态。

6. **可配置的感知-传输策略**
   
   支持多级触发模式：
   - 仅元数据（事件坐标、时间戳、置信度）
   - 关键帧+元数据
   - 完整视频流

---

## 技术方案

### 1. 整体架构

```
┌────────────────────────────────────────────────────────────┐
│                     系统架构示意图                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  光线入射方向                                               │
│      ↓                                                     │
│  ┌──────────────────────────────────────┐                  │
│  │          光学镜头 (Lens)              │                  │
│  ├──────────────────────────────────────┤                  │
│  │         微透镜阵列 (Microlens)        │                  │
│  ├─────────────────────┬────────────────┤                  │
│  │                     │                │                  │
│  │    背照式图像传感器  │                │  背面：光电二极管│
│  │    (BSI Sensor)     │                │  阵列（感光面）  │
│  │                     │                │                  │
│  │    ┌──────────────┐ │                │                  │
│  │    │ 光电二极管阵列│ │                │                  │
│  │    ├──────────────┤ │                │                  │
│  │    │ 金属互连层    │ │                │                  │
│  │    ├──────────────┤ │                │                  │
│  │    │ 逻辑电路层    │ │                │                  │
│  │    │ ┌──────────┐ │ │ ←─────────────┼── 正面：逻辑电路│
│  │    │ │并行数据  │ │ │               │                  │
│  │    │ │接口      │ │ │               │                  │
│  │    └─┬──────────┬─┘ │               │                  │
│  └──────┼──────────┼────┘               │                  │
│         │          │                    │                  │
│         │ Face-to-Face Hybrid Bonding   │                  │
│         │ (铜-铜直接键合)                │                  │
│         │          │                    │                  │
│  ┌──────┴──────────┴────────────────────┤                  │
│  │                                       │                  │
│  │    边缘感知处理芯片 (Edge Perceiver)   │  正面：逻辑电路  │
│  │                                       │                  │
│  │    ┌──────────────────────────┐      │                  │
│  │    │ 并行数据接收接口          │      │                  │
│  │    ├──────────────────────────┤      │                  │
│  │    │ 输入预处理 (Resize/Norm) │      │                  │
│  │    ├──────────────────────────┤      │                  │
│  │    │ 神经网络事件检测加速器    │      │                  │
│  │    │ ├─ RRAM Weight Array     │      │                  │
│  │    │ │  (500KB-2MB)            │      │                  │
│  │    │ ├─ Digital MAC Array     │      │                  │
│  │    │ │  (16×16 or 32×32)       │      │                  │
│  │    │ ├─ INT8/INT4 Quantized   │      │                  │
│  │    │ └─ Feature Buffer SRAM   │      │                  │
│  │    │    (64KB)                 │      │                  │
│  │    ├──────────────────────────┤      │                  │
│  │    │ 事件决策逻辑              │      │                  │
│  │    │ ├─ Softmax/Threshold     │      │                  │
│  │    │ ├─ 触发逻辑配置          │      │                  │
│  │    │ └─ 元数据生成             │      │                  │
│  │    ├──────────────────────────┤      │                  │
│  │    │ 可控高速传输接口 (HSMT)   │      │                  │
│  │    │ ├─ GMSL3 PHY             │      │                  │
│  │    │ ├─ FPD-Link IV PHY       │      │                  │
│  │    │ └─ 电源管理单元           │      │                  │
│  │    └──────────────────────────┘      │                  │
│  │                                       │                  │
│  └───────────────────────────────────────┘                  │
│                                                            │
│        ┌──────────────────────────────────┐                │
│        │      GMSL3/FPD-Link IV 线缆      │                │
│        │      (高速传输，事件触发)         │                │
│        └──────────────┬───────────────────┘                │
│                       │                                     │
│        ┌──────────────▼───────────────────┐                │
│        │     主处理器 (地平线Journey等)    │                │
│        │     ├─ 大算力NPU                 │                │
│        │     ├─ 复杂决策规划              │                │
│        │     └─ 全功能操作系统            │                │
│        └──────────────────────────────────┘                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 2. Face-to-Face键合详细设计

#### 与传统单芯片BSI传感器的区别

**现有技术（单芯片BSI CMOS）：**
```
┌───────────────────────┐
│  微透镜 / 彩色滤镜     │
├───────────────────────┤
│                       │
│  光电二极管阵列        │  ← 背部（Back Side）接收光子
│  （在硅片深层）        │
│                       │
├───────────────────────┤
│  金属互连层           │
│  （铜线布线）          │
├───────────────────────┤
│                       │
│  晶体管逻辑电路        │  ← 前部（Front Side）
│  （ADC、控制逻辑）     │      在同一die内完成信号处理
│                       │
└───────────────────────┘
         │
         ▼
    焊球/BGA封装
         │
         ▼
    PCB板上的连接器 → 外部处理芯片（通过MIPI/GMSL串行传输）
```

**本发明（双芯片Face-to-Face堆叠）：**
```
┌───────────────────────┐
│  微透镜 / 彩色滤镜     │
├───────────────────────┤
│                       │
│  光电二极管阵列        │  ← OV Sensor Die (第一芯片)
│                       │      BSI结构，光线从背面入射
├───────────────────────┤
│  金属互连层           │
├───────────────────────┤
│  晶体管逻辑电路        │
│  ┌─────────────────┐  │  ← 并行数据接口（数千条线）
│  │ 并行数据输出     │  │      通过Face-to-Face键合
│  └────────┬────────┘  │      直接连接到第二芯片
└───────────┼───────────┘
            │ Face-to-Face 混合键合
            │ (Hybrid Bonding)
            ▼
┌───────────────────────┐
│  ┌─────────────────┐  │
│  │ 并行数据接收     │  │  ← 边缘感知处理芯片 (第二芯片)
│  └─────────────────┘  │      独立的处理逻辑die
│  神经网络加速器        │
│  ├─ RRAM Weight Array │
│  ├─ Digital MAC Array │
│  └─ HSMT Controller   │
└───────────────────────┘
```

**关键创新：**
- 将传统**单芯片内的信号处理**拆分为**双芯片堆叠**
- 第一芯片（BSI传感器）仅保留光电转换和基础逻辑
- 第二芯片（处理芯片）集成RRAM存储和神经网络加速器
- 两芯片通过Face-to-Face键合实现数千条并行线连接，无需PCB走线和串行编码

#### 2.1 键合方式

**Hybrid Bonding（混合键合）**
- 间距（Pitch）：9μm 或 5μm（先进工艺）
- 材料：铜-铜直接键合 + 氧化物键合
- 优势：高密度、低电阻、高可靠性

**替代方案：Micro-bump + TSV**
- 如果Hybrid Bonding工艺不成熟，可使用微凸点+短TSV
- 间距：40-50μm
- 成本较低，但密度和性能稍差

#### 2.2 接口信号定义

```verilog
// 并行数据接口信号 (示例：2MP传感器)
interface PixelParallelInterface (
    input  logic clk_pixel,           // 像素时钟 (如 96MHz)
    input  logic [11:0] pixel_data,   // 12-bit 像素值 (RAW12)
    input  logic [10:0] x_addr,       // 水平地址
    input  logic [9:0]  y_addr,       // 垂直地址
    input  logic frame_valid,         // 帧有效
    input  logic line_valid,          // 行有效
    input  logic [7:0] sensor_id,     // 传感器ID (多传感器场景)
    output logic sensor_cfg_scl,      // I2C时钟 (配置)
    inout  logic sensor_cfg_sda,      // I2C数据 (配置)
    output logic sensor_reset_n,      // 复位
    output logic sensor_xmaster       // 主时钟使能
);
```

**信号数量估算：**
- 数据线：12-bit × 4 (quad pixel readout) = 48 lines
- 地址/控制：~20 lines
- 电源/地：~50 lines (分布式供电)
- 配置接口：2 lines
- **总计：~120条信号线**
- 加上冗余和测试点，实际键合点约 **200-300个**

### 3. 基于RRAM的神经网络事件检测引擎

#### 3.1 核心创新：RRAM存储 + 数字MAC计算

**传统方案的问题：**
```python
# 传统背景建模需要大容量存储
class TraditionalEventDetector:
    def __init__(self):
        # 需要存储2-3帧，约2MB SRAM
        self.frame_buffer = SRAM(2 frames)      # 900KB
        self.background_model = SRAM(1 frame)   # 450KB
        # 功耗高、面积大
```

**本发明方案（RRAM + CNN）：**
```python
# 使用训练好的CNN，权重存RRAM
class RRAM_CNN_EventDetector:
    def __init__(self):
        # RRAM只存权重，无需大容量帧缓存
        self.weight_array = RRAM_Array(500KB)   # 非易失，出厂前写入
        self.feature_buffer = SRAM(64KB)        # 仅需特征缓存
        self.mac_array = MAC_Array(32, 32)      # 数字MAC计算
        # 功耗<10mW，面积小
```

#### 3.2 RRAM Weight Array 设计

**存储容量规划**
```
神经网络: MobileNetV2-Tiny (自定义)
输入分辨率: 320×240 (从640×480缩小)
网络结构:
  - Conv1: 3×3×16, stride 2    → 432 parameters
  - DWConv1: 3×3×16            → 144 parameters
  - PWConv1: 1×1×32            → 512 parameters
  - DWConv2: 3×3×32            → 288 parameters
  - PWConv2: 1×1×64            → 2,048 parameters
  - DWConv3: 3×3×64            → 576 parameters
  - PWConv3: 1×1×128           → 8,192 parameters
  - GlobalAvgPool
  - FC: 128 → 2 (softmax)      → 256 parameters
  
总参数量: ~12K (FP32) ≈ 48KB
INT8量化后: ~48KB
含偏置、BN参数: ~100KB

安全余量: 5x
RRAM容量: 500KB (4Mb)
```

**RRAM技术参数 (GF 22nm)**
- 单元尺寸: ~50F² (F=22nm, 单元面积 ~0.024µm²)
- 存储密度: 500KB需 ~12,000 µm² = 0.012mm²
- 读取功耗: <1µW/Mb @ 100MHz
- 写入功耗: ~100µW/cell (仅出厂前写入)
- 读取延迟: <50ns
- 保持时间: >10年 @ 85°C

**写入策略**
```python
class RRAM_Programming:
    """
    出厂前权重编程
    """
    def __init__(self):
        self.max_write_cycles = 1000  # RRAM支持1K-10K次写入
        
    def fine_tune_weights(self, pretrained_model, calibration_data):
        """
        出厂前微调权重
        - 从预训练模型开始
        - 使用目标场景数据校准
        - 将最优权重写入RRAM
        """
        # 1. 加载预训练模型
        model = load_model(pretrained_model)
        
        # 2. 使用calibration_data微调
        model.train(calibration_data, epochs=10)
        
        # 3. INT8量化
        quantized_weights = quantize_to_int8(model.weights)
        
        # 4. 写入RRAM
        for addr, weight in enumerate(quantized_weights):
            self.rram_array.write(addr, weight)
            
        # 5. 验证写入
        verify_ok = self.verify_rram(quantized_weights)
        return verify_ok
```

#### 3.3 数字MAC阵列加速器

**架构设计 (Option A: 全数字)**
```verilog
module CNN_Accelerator (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [31:0]  pixel_in,        // 4 pixels × 8bit
    input  logic         pixel_valid,
    output logic [15:0]  event_prob,      // Softmax输出
    output logic         event_valid
);

    // RRAM Weight Array (只读，推理阶段)
    rram_weight_array #(
        .DEPTH(128*1024),  // 128K × 32bit = 4Mb (512KB)
        .WIDTH(32)
    ) weight_mem (
        .clk(clk),
        .addr(weight_addr),
        .rdata(weight_data),
        .read_en(weight_read_en)
    );
    
    // Digital MAC Array (32×32并行)
    logic [7:0]  activation [0:31];       // INT8激活值
    logic [7:0]  weight     [0:31];       // INT8权重 (从RRAM读出)
    logic [15:0] mac_result [0:31];       // INT16结果
    
    genvar i;
    generate
        for (i = 0; i < 32; i++) begin : mac_gen
            mac_unit #(
                .A_WIDTH(8),
                .B_WIDTH(8),
                .OUT_WIDTH(16)
            ) mac (
                .a(activation[i]),
                .b(weight[i]),
                .out(mac_result[i])
            );
        end
    endgenerate
    
    // 累加树
    logic [15:0] sum_result;
    tree_adder #(
        .NUM_INPUTS(32),
        .DATA_WIDTH(16)
    ) adder_tree (
        .in(mac_result),
        .out(sum_result)
    );
    
    // 后处理：BatchNorm + ReLU + (可选Pooling)
    // 最后Softmax层输出事件概率
    
endmodule

// 单个MAC单元 (INT8×INT8=INT16)
module mac_unit (
    input  logic [7:0]  a,      // activation
    input  logic [7:0]  b,      // weight
    output logic [15:0] out     // product
);
    assign out = $signed(a) * $signed(b);
endmodule
```

**性能指标**
```
工艺: GF 22nm FDSOI
工作频率: 100MHz
MAC阵列: 32×32 = 1024 MACs/cycle
峰值算力: 1024 × 100MHz × 2 (INT8) = 204.8 GOPS (理论)
实际算力: ~10 GOPS (考虑数据复用、控制开销)

功耗估算:
- RRAM读取: 1µW/Mb × 4Mb = 4µW
- MAC阵列: 1024 MACs × 10µW/MAC @100MHz ≈ 10mW
- 控制逻辑 + SRAM缓冲: ~5mW
- 总计: <20mW (Active)
- Idle模式: <1mW (仅保持电路)
```

#### 3.4 推理流程

```python
class RRAM_CNN_Inference:
    """
    RRAM CNN推理引擎
    """
    
    def __init__(self, rram_weights):
        self.weight_array = rram_weights  # 512KB RRAM
        self.feature_buffer = SRAM(64KB)   # 特征图缓存
        self.mac_array = MAC_Array(32, 32)
        
    def detect_event(self, input_frame):
        """
        输入: 640×480 RGB帧
        输出: 事件概率 (0.0-1.0)
        延迟: <5ms
        功耗: <20mW
        """
        # Step 1: 预处理 (硬件)
        resized = resize(input_frame, 320, 240)
        normalized = normalize(resized)  # INT8: -128 to 127
        
        # Step 2: CNN推理 (逐层)
        feature_map = normalized
        for layer in network_layers:
            if layer.type == 'CONV':
                feature_map = self.conv_layer(
                    feature_map,
                    weights=self.rram_read(layer.weight_addr),
                    mac_array=self.mac_array
                )
            elif layer.type == 'DEPTHWISE_CONV':
                feature_map = self.dwconv_layer(
                    feature_map,
                    weights=self.rram_read(layer.weight_addr)
                )
            elif layer.type == 'FULLY_CONNECTED':
                logits = self.fc_layer(
                    feature_map,
                    weights=self.rram_read(layer.weight_addr)
                )
        
        # Step 3: Softmax
        event_prob = softmax(logits)[1]  # 类别1: 有事件
        
        return event_prob
    
    def conv_layer(self, input_fm, weights, mac_array):
        """
        卷积层计算
        - 从RRAM读取权重
        - 使用MAC阵列并行计算
        - 输出特征图写入SRAM缓冲
        """
        output_fm = []
        for oc in range(output_channels):
            for oy in range(output_height):
                for ox in range(output_width):
                    # 计算一个输出像素
                    accum = 0
                    for ky in range(kernel_h):
                        for kx in range(kernel_w):
                            for ic in range(input_channels):
                                # 并行读取权重和激活
                                w = self.rram_read(weight_addr)
                                a = input_fm[ic][oy+ky][ox+kx]
                                # MAC计算
                                accum += mac_array.compute(a, w)
                    
                    # 加偏置、BatchNorm、ReLU
                    output_fm[oc][oy][ox] = relu(batchnorm(accum + bias))
        
        return output_fm
```

#### 3.5 功耗优化技术

**1. 动态电压频率调节 (DVFS)**
   - Idle模式：0.5V，1MHz，功耗 <1mW (仅监控电路)
   - Active模式：0.8V，100MHz，功耗 <20mW (CNN推理)
   - 仅在事件检测时切换到Active模式

**2. 时钟门控**
   - RRAM读取电路：仅在需要权重时开启
   - MAC阵列：仅在计算时开启，空闲时完全关闭
   - HSMT接口：无事件时完全断电

**3. 电源域划分**
   ```
   Power Domain 0 (Always-on): 
   - 像素接口接收
   - RRAM待机保持
   - 事件触发监控逻辑
   - 功耗: <1mW
   
   Power Domain 1 (Switchable):
   - CNN加速器 (RRAM读取 + MAC阵列)
   - 激活时功耗: ~15mW
   
   Power Domain 2 (Switchable):
   - HSMT高速传输接口
   - 仅事件触发时上电
   - 功耗: 50-200mW (active)
   ```

**4. 权重压缩与稀疏化**
   - 使用INT4/INT8混合精度
   - 权重剪枝：移除不重要的连接
   - 稀疏计算：跳过零值权重
   - 可减少30-50%计算量和功耗

### 4. 可控高速传输接口 (HSMT)

#### 4.1 架构

```verilog
module ControllableHSMT (
    input  logic        clk,
    input  logic        rst_n,
    
    // 来自CNN事件检测的触发信号
    input  logic        event_trigger,
    input  EventInfo    event_info,
    
    // 帧缓存接口
    input  FrameBuffer  frame_buffer,
    
    // GMSL3/FPD-Link IV PHY 接口
    output logic        hsmt_tx_data,
    output logic        hsmt_tx_clk,
    input  logic        hsmt_rx_data,  // 用于配置和控制
    
    // 电源管理
    output logic        hsmt_power_en,
    output logic        hsmt_pll_en
);

    // 状态机
    typedef enum logic [2:0] {
        STATE_IDLE,           // 关闭状态，功耗≈0
        STATE_WAKEUP,         // 唤醒PHY，PLL锁定 (2-5ms)
        STATE_TRANSMIT,       // 传输数据
        STATE_COOLDOWN,       // 传输后冷却
        STATE_SHUTDOWN        // 关闭PHY
    } hsmt_state_t;
    
    hsmt_state_t state, next_state;
    
    // 触发后行为
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        case (state)
            STATE_IDLE: 
                next_state = event_trigger ? STATE_WAKEUP : STATE_IDLE;
                
            STATE_WAKEUP:
                next_state = pll_locked ? STATE_TRANSMIT : STATE_WAKEUP;
                
            STATE_TRANSMIT:
                next_state = transmission_complete ? STATE_COOLDOWN : STATE_TRANSMIT;
                
            STATE_COOLDOWN:
                next_state = (cooldown_counter == 0) ? STATE_SHUTDOWN : STATE_COOLDOWN;
                
            STATE_SHUTDOWN:
                next_state = STATE_IDLE;
        endcase
    end
    
    // 功耗控制
    assign hsmt_power_en = (state != STATE_IDLE);
    assign hsmt_pll_en   = (state == STATE_WAKEUP || state == STATE_TRANSMIT || state == STATE_COOLDOWN);
    
endmodule
```

#### 4.2 传输模式配置

```python
class TransmissionConfig:
    """
    可配置的传输策略
    """
    
    # 模式1: 仅元数据 (最低功耗)
    MODE_METADATA_ONLY = 1
    # 传输: timestamp + bbox + confidence + event_type
    # 数据量: ~100 bytes
    # 传输时间: <1ms
    # HSMT激活时间: 最短
    
    # 模式2: 关键帧+元数据
    MODE_KEYFRAME_METADATA = 2
    # 传输: 元数据 + 缩略图 (如 320x240 JPEG)
    # 数据量: ~50KB
    # 传输时间: ~10ms
    
    # 模式3: 完整视频流 (传统模式)
    MODE_FULL_STREAM = 3
    # 传输: 30fps 完整视频
    # 数据量: ~10Mbps
    # 传输时间: 持续到事件结束
    
    # 模式4: 自适应 (推荐)
    MODE_ADAPTIVE = 4
    # 首次触发: MODE_METADATA_ONLY
    # 主SOC确认重要: 升级到 MODE_KEYFRAME_METADATA
    # 持续活动: 保持 MODE_FULL_STREAM
```



---

## 有益效果

### 1. 功耗对比

| 场景 | 传统架构 | 本发明架构 | 节省比例 |
|-----|---------|-----------|---------|
| Idle (无事件) | 2-5W | 5-20mW | **99%+** |
| Active (事件检测) | 10-20W | 0.1-0.3W | **98%+** |
| Active (事件处理+传输) | 10-20W | 1-3W | **80%+** |
| 平均功耗 (10%事件率) | 5-8W | 0.1-0.3W | **97%+** |

**功耗优势分析：**
- **RRAM vs SRAM**: RRAM读取功耗仅1µW/Mb，而SRAM动态功耗约100µW/Mb，降低100倍
- **CNN vs 背景建模**: 无需存储多帧图像，节省>2MB SRAM（约200mW功耗）
- **事件触发**: 99%时间处于<5mW idle模式

### 2. 延迟对比

| 环节 | 传统架构 | 本发明架构 | 提升 |
|-----|---------|-----------|------|
| 图像采集到事件检测 | 50-100ms | 1-5ms | **10-100x** |
| 事件检测到主SOC通知 | 100-200ms | 5-10ms | **10-20x** |
| 端到端总延迟 | 150-300ms | 10-20ms | **10-30x** |

### 3. 存储与面积优势

| 指标 | 传统方案 (SRAM) | 本发明 (RRAM) | 优势 |
|-----|----------------|--------------|------|
| 事件检测存储容量 | 2-4MB SRAM | 500KB RRAM | **4-8x减少** |
| 存储功耗 | 100-200mW | 1-5mW | **20-40x降低** |
| 芯片面积 (存储部分) | 2-4mm² | 0.1-0.2mm² | **10-20x减少** |
| 非易失性 | 否 (需持续供电) | 是 (掉电保持) | 可靠性提升 |

### 4. 智能化优势

| 特性 | 传统CV算法 | 本发明 (CNN) | 优势 |
|-----|-----------|-------------|------|
| 适应性 | 固定参数 | 可训练、可微调 | 场景自适应 |
| 检测精度 | 依赖阈值调参 | 数据驱动优化 | 精度提升10-20% |
| 误报率 | 较高 | 较低 | 减少50%+ |
| 功能扩展 | 需重新设计算法 | 重新训练即可 | 灵活性高 |

### 5. 系统集成度

- **封装尺寸**：与传统摄像头模组相当（约 15mm × 15mm）
- **外部元件**：仅需电源和高速接口连接器
- **即插即用**：兼容现有GMSL/FPD-Link车载协议
- **软件兼容**：主SOC无需修改驱动，仅接收事件数据
- **可配置性**：出厂前可通过RRAM编程适配不同场景

---

## 附图说明

### 图1: 传统架构vs本发明架构对比图
```
[左侧] 传统串行传输架构 (高功耗、高延迟、大存储)
[右侧] Face-to-Face + RRAM堆叠架构 (低功耗、低延迟、小存储)
[中间] 箭头标注关键改进点:
       - RRAM替代大容量SRAM
       - CNN替代传统背景建模
       - Face-to-Face键合替代PCB走线
```

### 图2: Face-to-Face键合 + RRAM集成截面图
```
[从上到下]
光学镜头 → 微透镜 → OV传感器BSI层 → 铜-铜键合 → 处理芯片逻辑层
                                                      ↓
                                                RRAM Weight Array
                                                      ↓
                                                Digital MAC Array
                                                      ↓
                                                HSMT Controller

标注：
- 光线入射方向（红色箭头）
- 像素数据流向（蓝色箭头）
- RRAM存储位置（绿色高亮）
- 数字MAC计算单元（橙色高亮）
- 键合点位置（黄色高亮）
```

### 图3: RRAM CNN事件检测引擎流程图
```
[输入] 640×480像素流 
    ↓
[预处理] Resize to 320×240 + Normalization
    ↓
[RRAM] 读取CNN权重 (500KB)
    ↓
[MAC阵列] 卷积计算 (CONV → DWCONV → PWCONV × N)
    ↓
[特征缓冲] SRAM缓存 (64KB)
    ↓
[全连接层] FC → 2 classes
    ↓
[Softmax] 输出事件概率
    ↓
[阈值判断] 如果 P(event) > 0.8 → 触发!
    ↓
[输出] 触发信号 + 元数据 / 可选图像数据
```

### 图4: RRAM Weight Array架构图
```
RRAM阵列结构:
┌─────────────────────────────────────┐
│  Wordline Decoder                   │
├─────────────────────────────────────┤
│  ┌─────┬─────┬─────┬─────┐         │
│  │Cell │Cell │ ... │Cell │  × 1024 │  
│  │(4b) │(4b) │     │(4b) │         │  
│  ├─────┼─────┼─────┼─────┤         │  
│  │ ... │     │     │ ... │  × 1024 │  共1M cells = 4Mb = 512KB
│  ├─────┼─────┼─────┼─────┤         │  
│  │Cell │Cell │ ... │Cell │  × 1024 │  
│  └─────┴─────┴─────┴─────┘         │
├─────────────────────────────────────┤
│  Bitline Drivers + Sense Amps       │
├─────────────────────────────────────┤
│  Weight Output (INT8: 2 cells × 4b) │
└─────────────────────────────────────┘

写入控制 (出厂前):
- Programming Circuit
- Verify Circuit
- ECC Correction

读取控制 (运行时):
- Read-only mode
- <1µW/Mb static power
```

### 图5: 可控HSMT状态机与功耗曲线
```
状态转换图:
IDLE (5mW) → WAKEUP (50mW, 2-5ms) → TRANSMIT (200mW) → COOLDOWN (50mW, 10ms) → SHUTDOWN → IDLE

时间轴上的功耗曲线:
    200mW │              ┌──────┐
         │              │      │
     50mW │    ┌──────┐ │      │ ┌──────┐
         │    │      │ │      │ │      │
      5mW ├────┘      └─┘      └─┘      └──────
         └──────────────────────────────────────
              ↑      ↑       ↑       ↑
           事件触发  PLL锁定 传输完成 冷却结束
```

### 图6: 多传感器扩展架构
```
单芯片支持多sensor die的拓扑结构:

Chiplet 0: OV Sensor Die + RRAM-CNN Die (Primary)
Chiplet 1: OV Sensor Die + RRAM-CNN Die (Secondary)
Chiplet 2: OV Sensor Die + RRAM-CNN Die (Tertiary)
         ...
         
共享HSMT总线 (按需仲裁)
```

---

## 具体实施方式

### 实施例1: 车载前视摄像头应用

**应用场景**
- L2+辅助驾驶系统前视主摄像头
- 分辨率：2MP (1920×1080)
- 帧率：30fps
- 主SOC：地平线Journey 6

**实现细节**
1. **Sensor选型**：OmniVision OV2311 (BSI, 2MP)
2. **Face-to-Face键合**：
   - 使用Hybrid Bonding工艺，9μm pitch
   - 键合点：约250个（含信号、电源、测试点）
   
3. **神经网络设计**：
   - 网络架构：MobileNetV2-Tiny (自定义)
   - 输入：320×240 (从1920×1080缩放)
   - 权重：500KB (INT8量化)
   - 训练：使用车载场景数据集预训练
   - Fine-tune：出厂前使用目标车型数据微调
   
4. **RRAM编程**：
   - 在GF 22nm fab进行weights写入
   - 使用专用测试机台
   - 写入后进行高温老化测试 (HTOL)
   - 验证读取稳定性
   
5. **触发策略**：
   - CNN输出P(event) > 0.8时触发
   - 首次触发：MODE_METADATA_ONLY
   - 主SOC请求后：MODE_FULL_STREAM
   - 无事件30秒后：返回STATE_IDLE
   
6. **性能指标**：
   - Idle功耗：5mW
   - 事件检测功耗：20mW
   - Active功耗：800mW (含HSMT)
   - 事件检测延迟：<5ms
   - 传输延迟：<10ms
   - 检测准确率：>95% (vs 传统算法 85%)

### 实施例2: 机器人视觉避障

**应用场景**
- 服务机器人360°环境感知
- 6路摄像头（前/后/左/右/上/下）
- 分辨率：1MP (1280×720)
- 帧率：15fps

**实现细节**
1. **多Sensor集成**：
   - 单芯片集成6个OV9281 sensor die
   - 通过时分复用轮流处理
   - 每路摄像头独立RRAM权重 (可配置不同检测目标)
   
2. **神经网络设计**：
   - 网络架构：轻量版YOLO-Tiny
   - 输入：320×240
   - 检测目标：人、宠物、障碍物
   - 权重：800KB per camera
   - 推理时间：15ms
   
3. **Fine-tune策略**：
   - 家庭场景：检测人、宠物
   - 仓库场景：检测货架、叉车
   - 户外场景：检测行人、车辆
   
4. **触发策略**：
   - 仅检测到危险障碍物时触发
   - 传输：元数据 + 1fps关键帧
   - 紧急事件立即触发 (延迟<20ms)
   
5. **性能指标**：
   - 平均功耗：30mW per camera (180mW总计)
   - 检测准确率：>92%
   - 误报率：<0.1次/小时
   - 紧急响应延迟：<20ms

### 实施例3: 智能监控摄像头 (电池供电)

**应用场景**
- 户外智能监控
- 分辨率：4MP (2560×1440)
- 要求：7×24小时运行，电池供电

**实现细节**
1. **超低功耗优化**：
   - 传感器进入低帧率模式（1fps）用于idle监控
   - 检测到事件后提升到30fps
   - RRAM静态功耗极低 (<1µW/Mb)
   
2. **太阳能供电**：
   - 平均功耗：<20mW
   - 可由小型太阳能板 (5W) + 电池供电
   - 阴天可续航72小时
   
3. **神经网络设计**：
   - 网络架构：二分类CNN (人/车 vs 背景)
   - 权重：300KB
   - 输入：160×120 (极低分辨率用于idle检测)
   - 检测到后切换到640×480详细分析
   
4. **事件过滤**：
   - 过滤风吹草动等自然运动
   - 仅传输人/车事件
   - 支持入侵检测、越界检测
   
5. **性能指标**：
   - Idle功耗：<10mW
   - 检测功耗：15mW
   - 传输功耗：200mW (事件发生时)
   - 平均功耗：12mW (假设1%事件率)
   - 续航：太阳能+电池可支持365天运行

---

## 权利要求书建议（简要）

1. **一种基于背照式图像传感器面对面键合与阻变存储器的视觉感知芯片架构**，其特征在于，包括：
   - 背照式图像传感器，光线从背面入射，逻辑电路位于正面；
   - 边缘感知处理芯片，通过面对面键合方式与所述图像传感器的正面直接连接；
   - 所述面对面键合采用铜-铜直接键合或微凸点键合；
   - 所述边缘感知处理芯片内部集成阻变存储器（RRAM）阵列用于存储神经网络权重，以及数字MAC阵列加速器用于执行神经网络推理；
   - 所述神经网络用于检测视觉场景中的事件。

2. **根据权利要求1所述的架构**，其特征在于，所述面对面键合实现像素级并行数据接口，直接传输未经串行编码的像素数据。

3. **根据权利要求1所述的架构**，其特征在于，所述阻变存储器（RRAM）阵列的权重在出厂前通过有限次编程操作进行配置，在推理阶段为只读模式。

4. **根据权利要求1所述的架构**，其特征在于，所述数字MAC阵列加速器采用全数字计算架构，支持INT8或INT4量化推理。

5. **根据权利要求1所述的架构**，其特征在于，所述阻变存储器（RRAM）阵列容量为100KB-2MB，存储卷积神经网络的权重参数。

6. **根据权利要求1所述的架构**，其特征在于，还包括可控高速传输接口，仅当所述神经网络检测到有效事件时才激活向主处理器的传输。

7. **根据权利要求1所述的架构**，其特征在于，支持多传感器扩展，在同一封装内集成多个图像传感器die，每个传感器对应独立的RRAM权重存储。

8. **根据权利要求1所述的架构**，其特征在于，所述芯片采用GF 22nm FDSOI工艺制造，所述RRAM与逻辑电路集成在同一芯片上。

...（可根据需要补充更多权利要求）

---

## 市场应用前景

### 目标市场

1. **车载视觉系统**（最大市场）
   - 全球ADAS摄像头市场：2025年预计 150亿美元
   - 前装量产车型需求量大
   - 功耗敏感（电动车续航）
   - RRAM方案的极低功耗优势显著

2. **机器人视觉**（高增长市场）
   - 服务机器人、AGV、无人机
   - 对成本和功耗极度敏感
   - 国内供应链成熟
   - 可配置CNN适应不同场景

3. **智能监控**（存量市场）
   - 太阳能供电监控
   - 电池供电野外监控
   - 边缘AI相机
   - 存量替换需求

4. **工业视觉**（专业市场）
   - 产线质量检测
   - 缺陷检测
   - 可针对不同产品fine-tune CNN

### 竞争优势

1. **技术壁垒**：
   - Face-to-Face BSI 堆叠技术门槛高
   - RRAM+CNN架构独特
   - 非易失存储+超低功耗组合

2. **专利保护**：
   - 核心架构可申请多项发明专利
   - RRAM在视觉检测中的应用
   - 出厂前fine-tune方法

3. **成本优势**：
   - 长期看可减少外部存储和PCB面积
   - RRAM面积比SRAM小10-20倍
   - 低功耗减少散热成本

4. **性能优势**：
   - 检测精度比传统CV算法高10-20%
   - 功耗降低90%+
   - 可配置适应不同场景

5. **生态兼容**：
   - 兼容现有GMSL/FPD-Link协议
   - 无需改动主SOC软件
   - 可替换传统摄像头模组

### 商业模式建议

1. **芯片销售**：面向模组厂商销售芯片
2. **IP授权**：将Face-to-Face+RRAM架构授权给sensor厂商
3. **模组销售**：直接生产智能摄像头模组
4. **定制服务**：针对特定场景提供fine-tune服务
5. **系统方案**：与地平线、NVIDIA等主SOC厂商深度合作

---

## 专利申请策略建议

### 专利布局建议

1. **核心架构专利**（本交底书）
   - 优先权：立即申请
   - 地域：中国、美国、欧洲、日本
   
2. **制造工艺专利**
   - Face-to-Face键合工艺流程
   - RRAM与CMOS集成方法
   - 混合键合优化方法
   
3. **算法与架构专利**
   - RRAM存储CNN权重的视觉检测架构
   - 出厂前fine-tune方法
   - 超低功耗事件检测算法硬件实现
   - INT8/INT4量化推理优化
   
4. **应用专利**
   - 特定场景的优化实施例
   - 多传感器扩展方法
   - 自适应触发策略

### 注意事项

1. **保密**：在申请日前严格保密，避免公开披露
2. **优先权**：尽快准备正式申请文件（建议6个月内）
3. **现有技术检索**：申请前进行充分的专利检索
   - 特别关注Sony、Samsung的3D堆叠专利
   - 关注RRAM在AI加速器的应用专利
4. **合作方协议**：与OmniVision、GF等潜在合作方签署NDA
5. **RRAM专利**：确认GF 22nm RRAM工艺的专利授权情况

---

## 附录

### A. 相关技术标准
- MIPI CSI-2 Specification
- GMSL3 (Maxim/ADI) Specification
- FPD-Link IV (TI) Specification
- ISO 26262 (汽车功能安全)
- AEC-Q100 (汽车电子可靠性)
- IEEE 2851 (RRAM标准，如有)

### B. 参考芯片规格
- Sony IMX series BSI sensors
- OmniVision OV series BSI sensors
- 地平线Journey系列SOC
- NVIDIA Drive系列SOC
- Mythic AI (RRAM-based AI chip)
- Weebit Nano RRAM IP

### C. 制造工艺参考
- GF 22nm FDSOI Platform
- GF RRAM IP (与Adesto合作)
- TSMC 22ULL RRAM
- 日月光/长电科技 3D封装工艺
- Intel EMIB (Embedded Multi-die Interconnect Bridge)

### D. 神经网络参考架构
- MobileNetV2
- EfficientNet-Lite
- YOLO-Tiny
- 自定义轻量级CNN

---

**文档版本**: v2.0 (RRAM Edition)
**撰写日期**: 2026-02-26
**技术联系人**: [待填写]
**法律联系人**: [待填写]

---

*本交底书仅供内部讨论使用，未经许可不得对外披露*

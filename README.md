# Lawyer LLaMA

通过指令微调，LLaMa 模型在通用领域展现出了非常好的表现。但由于缺少合适的数据，少有人探究LLaMa在法律领域的能力。为了弥补这一空白，我们提出了Lawyer LLaMA，一个在法律领域数据上进行了额外训练的模型。

Lawyer LLaMA 首先在大规模法律语料上进行了continual pretraining，让它系统的学习中国的法律知识体系。
在此基础上，我们借助ChatGPT收集了一批对中国国家统一法律职业资格考试客观题（以下简称法考）的分析和对法律咨询的回答，利用收集到的数据对模型进行指令微调，让模型习得将法律知识应用到具体场景中的能力。

我们的模型能够：

1. **掌握中国法律知识：** 能够正确的理解民法、刑法、行政法、诉讼法等常见领域的法律概念。例如，掌握了刑法中的犯罪构成理论，能够从刑事案件的事实描述中识别犯罪主体、犯罪客体、犯罪行为、主观心理状态等犯罪构成要件。模型利用学到的法律概念与理论，能够较好回答法考中的大部分题目。

2. **应用于中国法律实务**：能够以通俗易懂的语言解释法律概念，并且进行基础的法律咨询，涵盖婚姻、借贷、海商、刑事等法律领域。

为了给中文法律大模型的开放研究添砖加瓦，本项目将开源一系列**法律领域的指令微调数据**和**基于LLaMA训练的中文法律大模型的参数** 。

## 新闻

[2023/4/13] 开源了由ChatGPT生成的指令微调数据，包含2k条法考题目的解答及5k条法律咨询回复。

## 更新计划

- [ ] 开源更大规模的指令微调数据

- [ ] 开源法律咨询多轮对话数据

- [ ] 开源通用法律领域的预训练模型

- [ ] 开源特定法律领域的咨询模型

## 训练数据

### 法律领域纯文本语料

我们收集了部分公开的法律数据用于模型的continual training，包括法律条文、裁判文书等。

### 通用指令微调数据

1. [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 52k 英文。

2. [BELLE](https://github.com/LianjiaTech/BELLE) 52K 中文。

BELLE提供了2M的中文指令微调数据，我们从中随机采样了52K数据，和Alpaca混合后，用于通用领域的指令微调。

### 法律指令微调数据

#### 法考数据
我们将[JEC-QA中国法考数据集](https://jecqa.thunlp.org/)中的试题输入ChatGPT，让其生成对每个答案的解析。

    Input:

    根据《物权法》，下列关于物权变动的特别规定，表述正确吗？
    因继承或者受遗赠取得物权的，自继承或者受遗赠开始时发生效力
    请给出详细的推理过程之后再给出答案。

    Output:

    这个表述是正确的。
    根据《物权法》第X条规定：“因继承或者受遗赠取得物权的，自继承或者受遗赠开始时发生效力。”这意味着，当一个人通过继承或受遗赠获得物权时，这个物权的效力是从继承或受遗赠开始时就生效的，而不是从物权转移的时间开始生效。这是因为，继承或受遗赠是一种法定的方式来获得物权，因此，物权的效力应该从这个法定的时间点开始生效。

#### 法律咨询
我们从[开放数据集](https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/OLO4G8)中收集了一些咨询问题，输入ChatGPT，让ChatGPT扮演律师解答问题。

    Input:

    是这么个回事，关于我儿子今年2月27日背着我们大人买了一台轻便女式摩托车麻烦你帮忙看怎么解决？

    Output:

    首先，根据法律规定，未满18岁的未成年人是无法独立签订合同的，因此你的儿子在未经你的同意的情况下购买摩托车是无效的。
    其次，根据《中华人民共和国道路交通安全法》的规定，未满18周岁的人不得驾驶摩托车，因此你的儿子也无权驾驶这辆摩托车。
    建议你尽快联系卖家，说明情况并要求退货。如果卖家不同意，你可以向消费者协会或相关部门投诉，寻求法律帮助。同时，你也需要对你的儿子进行教育，让他了解道路交通安全法规，避免再次发生类似的情况。

## 模型参数

在符合LLaMA使用条款的情况下，我们将于近期公开基于LLaMA 7B训练的中文法律模型。

## **系统效果**

我们选取了LLaMA 7B作为基座模型，使用法律领域语料进一步预训练，并在通用领域和法律领域指令微调。

以下对比了该模型和ChatGPT对于部分输入样例的回复：

<details>
<summary><b>法律概念理解</b></summary>

**Input:** 什么是委托代理？

| **Lawyer LLaMA (7B)**                                                                                                           | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 委托代理是指委托人将自己的权利和义务委托给代理人，代理人代表委托人进行某项法律行为。委托代理是一种法律关系，代理人必须按照委托人的要求行事，并向委托人报告进展情况。委托代理的法律效力与委托人的权利和义务相同，代理人必须遵守法律法规，保护委托人的合法权益。 | 委托代理是指一方（即委托人）授权另一方（即代理人）代表其执行特定的法律行为或采取特定的法律行动。委托代理通常涉及商业或法律事务，例如买卖房屋或签订合同等。在这种情况下，委托人必须给代理人明确的授权，以便代理人能够代表委托人采取行动。<br/>委托代理可以是口头协议或书面协议，具体取决于委托人和代理人之间的协议。委托代理需要遵守一定的法律规定，例如代理人必须履行忠实、诚信的义务，不能利用委托代理行为牟取不当利益或泄露委托人的保密信息等。<br/>委托代理在商业和法律事务中很常见，因为它可以让委托人将特定的任务交给代理人去完成，从而减轻自己的负担和节约时间和资源。同时，它也可以为代理人提供商业机会，例如代表多个客户进行买卖房屋或签订合同等。 |

---

**Input:** 什么是夫妻共同财产？

| **Lawyer LLaMA (7B)**                                                                                                           | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 夫妻共同财产是指夫妻在婚姻关系存续期间所得的财产，包括工资、奖金、房产、存款、股票等。这些财产是夫妻共同所有的，无论是谁来得的，都应该被视为夫妻共同财产。在离婚时，夫妻应该协商如何分配这些财产，如果无法协商，法院将会根据具体情况进行裁决。 | 夫妻共同财产是指夫妻在婚姻关系中共同拥有的财产，包括在婚姻关系存续期间所获得的财产和财产所产生的收益，以及婚姻关系存续期间夫妻个人所持有的财产中的共同财产份额。夫妻共同财产制度是指在婚姻关系中，夫妻共同财产的所有权属于夫妻双方，且需要夫妻双方共同管理、共同支配和共同分割的制度。在夫妻离婚、丧失中国国籍等情况下，夫妻共同财产应当依法进行分割。在中国，夫妻共同财产制度适用于所有婚姻关系存续期间所取得的财产，除非夫妻双方事先约定适用其他财产制度。|

---

**Input:** 什么是抵押？

| **Lawyer LLaMA (7B)**                                                                                                           | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 抵押是指将某种财产作为担保，以保证借款人在约定的时间内履行还款义务。抵押的财产可以是房产、车辆、股票、债券等。当借款人未能按时还款时，抵押人可以通过法律程序依法行使抵押权，即通过法院的强制执行程序，将抵押的财产变卖，以偿还借款人的债务。 | 抵押是指把财产或财产权利作为担保物，用于向银行或其他借款人借款的行为。抵押通常用于获得较大额度的贷款，例如购买房屋或商业物业的贷款。在抵押贷款中，房屋或其他资产通常被抵押，这意味着如果贷款未能按时还清，银行或借款人可以通过法律程序将抵押物出售以清偿债务。在一些国家，抵押可能需要登记或公证，以确保抵押权的优先性和合法性。抵押权是借款人对抵押财产的权利，而抵押物则是被抵押的财产或财产权利。|

---

**Input:** 什么是正当防卫？

| **Lawyer LLaMA (7B)**                                                                                                           | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 正当防卫是指在遭受非法侵害或者遭受即将发生非法侵害的情况下，为了保护自己或他人的人身、财产安全，采取必要的防卫行为，造成非法侵害人的行为受到制止或者阻止的情况。正当防卫是一种合法的防卫行为，但是在采取必要的防卫行为时，必须遵守必要性原则，即防卫行为必须是必要的、适当的、合理的、合法的。如果防卫行为超出了必要性原则，就可能构成过度防卫，属于违法行为。 | 正当防卫是指在保护自己、他人或者国家、集体的合法权益时，对正在非法侵犯自己等权益的侵害者采取必要的防卫行为的行为，具有法律上的正当性和合法性。<br/>正当防卫是许多国家刑法中都规定的一种法律制度，允许人们在一定的情况下采取必要的防卫行为，以保护自己、他人或者国家、集体的合法权益。在正当防卫的情况下，防卫人不承担任何刑事责任。<br/>正当防卫的条件包括：必要性、即时性和合理性。必要性指的是防卫行为必须是为了防止或制止正在进行的非法侵犯行为，且必须采取适当的方式和手段。即时性指的是防卫行为必须是在非法侵犯行为正在进行或即将发生时立即采取的。合理性指的是防卫行为必须是在保护自己、他人或者国家、集体的合法权益的情况下进行的，不得超过必要的限度。<br/>需要注意的是，正当防卫不是自由行为的理由，而是在必要的情况下为了保护合法权益采取的一种必要行为，因此必须符合法律规定的条件才能被认定为正当防卫。如果防卫行为不符合法律规定，可能会被认定为过当防卫，需要承担相应的法律责任。|

</details>

<details>
<summary><b>刑事案件中的犯罪构成理论 -- 四要件</b></summary>

**Input:** 请抽取以下案例中的犯罪构成要件 ‘绩溪县人民检察院起诉书指控：2013年11月1日8时40分许，被告人曹某某驾驶皖P54611自卸货车行至家朋乡汪店村阳戈溪村河边装石块，被告人曹某某将货车停放于一斜坡处，拉上手刹下即下车去驾驶铲车往货车内装石块，约二十分钟后货车溜车下滑，滑行中撞上在坡底河边洗衣服的洪某某，当日下午，洪某某经抢救无效死亡。经鉴定，洪某某系因事故致创伤性失血性休克死亡。’

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                     | **ChatGPT (175B)**                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 犯罪构成要件：<br/>1. 犯罪主体：被告人曹某某。<br/>2. 犯罪行为：驾驶货车在卸货行驶过程中，将货车停放于一斜坡处，拉上手刹下即下车去驾驶铲车往货车内装石块，约二十分钟后货车溜车下滑，滑行中撞上在坡底河边洗衣服的洪某某，当日下午，洪某某经抢救无效死亡。<br/>3. 结果：洪某某死亡。<br/>4. 主观方面：被告人曹某某的行为是有意识的，即他知道自己的行为存在危险，但仍然故意滑行，撞上了洪某某。<br/>综上所述，被告人曹某某的行为构成过失杀人罪。 | 这是一起交通肇事案件，犯罪构成要件包括：<br/>肇事：被告人曹某某驾驶自卸货车行至家朋乡汪店村阳戈溪村河边装石块，将货车停放于一斜坡处，拉上手刹下即下车去驾驶铲车往货车内装石块，货车因此失去控制溜车下滑，最终撞上在坡底河边洗衣服的洪某某。<br/>重大事故：该起交通事故导致洪某某死亡，经鉴定，洪某某系因事故致创伤性失血性休克死亡。<br/>因此，被告人曹某某构成交通肇事罪，属于过失致人死亡罪。 |

---


**Input:** 请抽取以下案例中的犯罪构成要件 '公诉机关指控，被告人林某与其妻子曾某因琐事发生口角，并商量到京山县民政局婚姻登记处办理离婚手续。2014年11月3日11时许，被告人林某与其妻子曾某到京山县民政局婚姻登记处（位于京山县新市镇绿林路）办理离婚手续时，双方发生争吵。被告人林某将曾某叫到婚姻登记处旁边的工商银行后院商谈，商谈中又动起手来，曾某用手抓伤被告人林某面部，被告人林某将曾某按在地上，拿出事先携带的匕首将曾某面部、颈部等划伤4处。经鉴定，被害人曾某的损伤程度已构成轻伤一级。'

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                     | **ChatGPT (175B)**                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 根据以上案例中的犯罪构成要件，可以得出以下推理过程：<br/>1. 公诉机关指控：案件已经被公诉机关提起公诉。<br/>2. 犯罪主体：被告人林某和其妻子曾某。<br/>3. 犯罪行为：被告人林某在与其妻子曾某商谈时，将曾某叫到婚姻登记处旁边的工商银行后院商谈，商谈中又动起手来，曾某用手抓伤被告人林某面部，被告人林某将曾某按在地上，拿出事先携带的匕首将曾某面部、颈部等划伤4处。<br/>4. 犯罪构成要件：被告人林某的行为已经构成了故意伤害罪。<br/>具体来说，被告人林某的行为涉嫌故意伤害罪的构成要件包括：主观方面的故意和客观方面的行为。<br/>首先，被告人林某的主观方面的故意是指他明知自己的行为会对他人造成伤害，但仍然故意实施的行为。在本案中，被告人林某明知自己的行为会对他人造成伤害，因为他在商谈时动起手来，并使用匕首将曾某面部、颈部等划伤4处。<br/>其次，被告人林某的客观方面的行为是指他的行为已经达到了故意伤害的客观要件。在本案中，被告人林某的行为已经达到了故意伤害的客观要件，因为他的行为是有预谋的，他明知自己的行为会对他人造成伤害，但仍然实施了这些行为。<br/>因此，根据以上推理过程，可以得出结论：被告人林某的行为已经构成了故意伤害罪。 | 犯罪构成要件包括：<br/>1. 故意伤害罪：被告人林某使用匕首对妻子曾某进行攻击，导致曾某面部、颈部等部位受伤，已构成轻伤一级，符合故意伤害罪的构成要件。<br/>2. 暴力妨害家庭罪：被告人林某在与妻子商谈离婚事宜时，发生争吵并将妻子带到婚姻登记处旁边商谈，期间动起手来，并使用匕首进行攻击，已构成暴力妨害家庭罪的构成要件。 |

---

**Input:** 请抽取以下案例中的犯罪构成要件  '三门峡市湖滨区人民检察院指控：2015年10月11日20时43分左右，被告人荆某驾驶豫P×××××号重型自卸货车，沿三门峡市209国道东半幅自南向北行驶至209国道与五原路交叉口北50米处，碰撞沿209国道东半幅自北向南步行的被害人姚某，致姚某当场死亡。事故发生后，荆某驾车逃逸。经三门峡市公安交警支队事故处理大队认定，荆某负事故全部责任。公诉机关认为，被告人荆某的行为已构成××罪，请依法判处。'

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                     | **ChatGPT (175B)**                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 犯罪构成要件：<br/>1. 行为主体：被告人荆某<br/>2. 客体：交通肇事罪<br/>3. 客体构成要件：荆某驾驶豫P×××××号重型自卸货车，沿三门峡市209国道东半幅自南向北行驶至209国道与五原路交叉口北50米处，碰撞沿209国道东半幅自北向南步行的被害人姚某，致姚某当场死亡。<br/>4. 主观方面的犯罪意图：荆某在行驶过程中没有采取必要的措施避免事故发生，也没有及时停车并报警，反而逃逸。<br/><br/>综上所述，被告人荆某的行为已构成交通肇事罪。 | 案例中的犯罪构成要件如下：<br/>1. 时间地点：2015年10月11日20时43分左右，在三门峡市209国道与五原路交叉口北50米处发生。<br/>2. 车辆和行人：被告人荆某驾驶豫P×××××号重型自卸货车，沿209国道行驶，撞上沿209国道东半幅自北向南行走的被害人姚某。<br/>3. 后果：被害人姚某当场死亡。<br/>4. 逃逸：案发后，被告人荆某驾车逃逸。<br/>5. 责任：经三门峡市公安交警支队事故处理大队认定，荆某负事故全部责任。<br/>基于以上事实，可以初步认定被告人荆某的行为构成“交通肇事罪”或“交通肇事致死罪”。 |

---

**Input:** 请抽取以下案例中的犯罪构成要件 '三亚市城郊人民检察院指控，2013年7月25日1时许，被告人李某驾驶摩托车搭载罗某沿河东路逆向行驶，罗某见被害人林某持一挎包乘坐电动车沿正常方向行驶便提议实施抢夺，被告人李某减速等待被害人林某乘坐的电动车超过其二人驾驶的摩托车后便调转车头靠近被害人林某，二车行驶至河东路规划局路口处时，罗某伸手将被害人林某的挎包抢走，得手后二人驾车迅速逃往三亚大桥方向，被抢的挎包内有人民币现金400元以及三星手机一部（经鉴定，价值人民币2819元）。破案后，被抢夺的财物未能追回。'

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                     | **ChatGPT (175B)**                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 根据以上案例，可以得出以下犯罪构成要件：<br/>1. 罪名：抢夺罪<br/>2. 主体犯罪：被告人李某和罗某<br/>3. 客体犯罪：抢夺行为<br/>4. 主观方面犯罪：罪主体有意实施抢夺行为，但是未能完成抢夺行为<br/>5. 客观方面犯罪：抢夺行为是在被害人林某的财物领域内实施的，且抢夺行为是在被害人林某的财物领域内完成的<br/>6. 犯罪结果：被告人李某和罗某抢夺了被害人林某的挎包内有人民币现金400元以及三星手机一部（经鉴定，价值人民币2819元）<br/>综上所述，被告人李某和罗某构成抢夺罪。 | 根据案例，可以提取以下犯罪构成要件：<br/>1. 抢劫罪构成要件：<br/>被告人李某、罗某明知林某持有财物，即电动车和挎包，伺机实施抢夺。在被害人林某乘坐电动车行驶时，被告人李某驾驶摩托车追逐并靠近电动车。当电动车行驶到规划局路口时，罗某伸手将林某的挎包抢走，并迅速逃离现场。被抢夺的财物包括现金和手机，价值一定数额，构成抢劫罪的犯罪构成要件。<br/>1. 交通肇事罪构成要件：<br/>被告人李某驾驶摩托车搭载罗某逆向行驶，违反了道路交通安全法规，构成交通肇事罪的犯罪构成要件。|

</details>


<details>
<summary><b>法考</b></summary>

**Input:** 假设甲国为《保护文学艺术作品伯尔尼公约》的成员国，乙国为非成员国。依该公约的规定，下列作品可以享有国民待遇?

A. 甲国公民在甲国和乙国同时出版的文学作品

B. 乙国公民首先在甲国出版的文学作品

C. 在甲国有住所的乙国公民的文学作品

D. 乙国公民在乙国发表的文学作品

**Gold Answer:** A B C



| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                                     | **ChatGPT (175B)**                                                                                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A: 根据《保护文学艺术作品伯尔尼公约》的规定，成员国应该给予其他成员国的公民和非成员国的公民在成员国境内创作的文学艺术作品国民待遇。也就是说，如果一个人在甲国创作了一部文学作品，那么他在乙国出版该作品也应该享有甲国的版权保护。<br/>但是，如果该作品的作者是乙国的公民，那么他不属于甲国的版权保护范围。因此，如果甲国的公民在甲国和乙国同时出版了文学作品，那么该作品只能享有甲国的版权保护，而不能享有乙国的版权保护。<br/>因此，答案是：甲国公民在甲国和乙国同时出版的文学作品可以享有甲国的国民待遇。 | A: 根据《保护文学艺术作品伯尔尼公约》的规定，成员国应该对其他成员国的公民在该公约规定保护范围内的作品享有国民待遇。但对于非成员国的作品，不一定能享有国民待遇。<br/>根据题目的描述，甲国是《保护文学艺术作品伯尔尼公约》的成员国，而乙国是非成员国。因此，甲国公民在甲国和乙国同时出版的文学作品应该能够在甲国享有国民待遇，但在乙国不一定能够享有国民待遇。<br/>总之，该选项的判断是“部分正确，部分错误” |
| B: 根据《保护文学艺术作品伯尔尼公约》的规定，成员国应该给予其他成员国的公民和非成员国的公民在成员国境内创作的文学艺术作品国民待遇。这意味着，如果乙国公民在甲国创作了一部文学作品，那么该作品应该受到甲国的保护，并且甲国应该给予该作品国民待遇，即与甲国公民的作品享有同等的保护。<br/>因此，如果乙国公民在甲国出版了一部文学作品，那么该作品可以享有国民待遇。 | B: 根据《保护文学艺术作品伯尔尼公约》的规定，成员国应该给予其他成员国的作者和作品同等的保护待遇，即国民待遇原则。这意味着，如果一个国家的作品在另一个国家出版或发行，它应该得到与后者本国作品相同的保护待遇。<br/>然而，该公约并未规定非成员国的作品可以享有国民待遇。因此，根据伯尔尼公约的规定，乙国公民首先在甲国出版的文学作品不能享有国民待遇。<br/>因此，选项是错误的。 |
| C: 根据《保护文学艺术作品伯尔尼公约》的规定，成员国应该给予其他成员国的公民和非成员国的公民在成员国境内创作的文学艺术作品国民待遇。这意味着，如果乙国公民在甲国创作了一部文学作品，那么该作品应该受到甲国的保护，并享有与甲国公民的作品相同的待遇。<br/>因此，如果乙国公民在甲国有住所，那么他的在甲国创作的文学作品可以享有国民待遇。这是因为，乙国公民在甲国有住所，就像甲国公民在乙国有住所一样，他应该享有与甲国公民相同的权利和待遇。<br/>因此，答案是：乙国公民在甲国创作的文学作品可以享有国民待遇。 | C: 根据《保护文学艺术作品伯尔尼公约》的规定，其成员国应当向其他缔约国公民的作品，给予与本国公民作品同等的国民待遇。因此，在甲国中，如果有乙国公民居住并创作了文学作品，那么该作品应该可以享有甲国的国民待遇，即与甲国公民创作的作品受到相同的保护。<br/>然而，乙国并非该公约的成员国，因此乙国公民在自己的国家创作的作品可能无法得到与甲国公民作品相同的保护。此外，甲国可能还有其他国内法规定，限制外国公民的著作权保护范围，需要具体情况具体分析。<br/>因此，对于该选项的判断为：可能正确，但需要具体情况具体分析 |
| D: 根据《保护文学艺术作品伯尔尼公约》的规定，成员国应该给予其他成员国的公民和非成员国的公民在成员国境内发表的文学艺术作品国民待遇。这意味着，如果乙国公民在乙国发表了文学作品，那么该作品可以享有与乙国公民的作品相同的保护，包括获得版权和受到财产权利的保护。<br/>而对于乙国公民在乙国发表的文学作品，则不具备国民待遇，因为乙国不是《保护文学艺术作品伯尔尼公约》的成员国。但是，如果乙国公民在乙国发表的文学作品在乙国被认定为优先申请，那么该作品可以在乙国获得版权和财产权利保护。<br/>因此，答案是：乙国公民在乙国发表的文学作品不可以享有国民待遇，但如果该作品在乙国被认定为优先申请，那么可以在乙国获得版权和财产权利保护。 | D: 根据《保护文学艺术作品伯尔尼公约》的规定，成员国应当给予其他成员国的著作权人及其作品同自己国民的待遇。因此，甲国应该给予乙国公民在甲国发表的文学作品同自己国民的待遇。但是，乙国公民在乙国发表的文学作品是否能够享有国民待遇，则取决于乙国是否加入了该公约。<br/>因为题目中指出乙国为非成员国，因此该国的公民在该公约下并没有享有国民待遇的权利。所以，选项错误。 |

</details>


<details>
<summary><b>咨询</b></summary>

**Input:** 男婚内出轨，怎样告女方？？？？？？？

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                                                                                                   | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 如果男方在婚姻关系中出轨，女方可以通过以下方式来表达自己的不满和愤怒：<br/>1. 与男方沟通：女方可以与男方沟通，表达自己的不满和愤怒，并要求男方停止出轨行为。<br/>2. 寻求婚姻咨询：女方可以寻求专业的婚姻咨询师或心理医生的帮助，以了解如何处理自己的情绪和如何保护自己的权益。<br/>3. 寻求法律援助：女方可以寻求法律援助，了解自己的权利和法律程序，以便采取适当的行动来保护自己的权益。<br/>4. 提起离婚：如果男方的出轨行为已经严重影响了婚姻关系，女方可以考虑提起离婚。在离婚过程中，女方可以要求分割财产和抚养子女的权利。<br/>无论采取哪种行动，女方都应该保持冷静和理智，以便更好地保护自己的权益。 | 作为律师，我必须提醒您，婚内出轨是一项敏感的问题，需要认真考虑和权衡。如果您想要采取法律行动，以下是一些可能有用的信息：<br/>1. 确定出轨是否构成违法行为：在某些国家或地区，婚内出轨可能被视为违法行为，可能构成通奸罪或与婚姻法相关的罪行。您应该检查您所在地区的相关法律，以确定出轨是否构成违法行为。<br/>2. 收集证据：如果出轨构成违法行为，您需要收集证据来支持您的指控。这可能包括获取出轨方的通讯记录、照片、视频等物证，或者聘请私人侦探来获取证据。<br/>3. 寻求法律援助：如果您打算采取法律行动，最好咨询一位有经验的律师。律师可以帮助您理解相关的法律，并提供建议和帮助。<br/>请注意，采取法律行动可能会对您的家庭关系产生负面影响。在采取任何行动之前，请考虑一下这是否值得。您可能需要与配偶进行对话，并尝试解决问题，或者寻求婚姻咨询等其他形式的帮助来修复您的婚姻关系。 |

---

**Input:** 分居两年有两个孩子，大女儿10岁，儿子五岁，男方总对我用家庭暴力，导致我想离婚法院会怎么判？

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                                                                                                   | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 根据中国的婚姻法，如果你可以证明你的丈夫对你实施了家庭暴力，法院可能会判决你获得离婚。在这种情况下，法院可能会考虑以下因素：<br/>1. 是否存在家庭暴力行为：如果你能够证明你的丈夫对你实施了家庭暴力行为，法院可能会认定这一点。<br/>2. 是否存在婚姻破裂：如果你能够证明你和你的丈夫之间的婚姻已经破裂，法院可能会认定这一点。<br/>3. 是否存在孩子的利益：如果你能够证明你的孩子受到了伤害，法院可能会认为这一点。<br/>4. 是否存在其他证据：如果你能够证明其他证据表明你的丈夫对你实施了家庭暴力行为，法院可能会认定这一点。<br/>因此，如果你能够证明你的丈夫对你实施了家庭暴力行为，法院可能会判决你获得离婚。但是，具体的判决结果还需要根据你的案件具体情况来决定。 | 首先，家庭暴力是不合法的行为，法院通常会采取严厉的措施来保护受害者。在你的情况下，离婚可能是合理的选择，但最终的判断取决于许多因素，如你是否有足够的证据证明你的配偶对你施暴，以及你的孩子是否会受到伤害。<br/>在考虑离婚的情况下，法院通常会优先考虑孩子的福利。法院可能会下令给你拥有孩子的抚养权，而你的配偶可能需要支付子女抚养费和配偶赡养费。同时，如果有家庭暴力的证据，法院可能会下令保护令，禁止你的配偶接近你和你的孩子，以确保你和你的孩子的安全。<br/>总之，如果你有家庭暴力的证据并且孩子的福利得到了保障，离婚是可能的，并且法院可能会支持你的要求。如果你需要更多的帮助和建议，我建议你寻求当地的法律援助组织或律师的帮助。 |

---

**Input:** 饮酒驾驶私家小车，和三个朋友出去玩，过弯道的时候撞到路边的树，车子翻到路沟请问司机要负什么责任？

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                                                                                                   | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 作为司机，您需要承担一定的责任。根据《中华人民共和国道路交通安全法》的规定，饮酒驾驶是违法行为，应当承担相应的法律责任。如果您的小车撞到了路边的树，导致车子翻到路沟，您需要承担以下责任：<br/>1. 违反交通规则的责任：根据《中华人民共和国道路交通安全法》的规定，饮酒驾驶是违法行为，应当承担相应的法律责任。<br/>2. 车辆损失的责任：如果您的小车在翻到路沟后摔掉或者撞到了其他车辆，您需要承担相应的车辆损失责任。<br/>3. 人身伤害的责任：如果您的行为导致他人受到人身伤害，您需要承担相应的人身伤害责任。<br/>因此，作为司机，您需要认真对待交通安全，遵守交通规则，避免饮酒驾驶等违法行为，保护自己和他人的安全。 | 根据您提供的情况描述，司机可能面临以下法律责任：<br/>1. 刑事责任：如果司机在事故发生时饮酒超过了法定限制，他可能会被控以醉酒驾车罪，这是一项刑事犯罪行为，可能会被判处罚款、监禁或者吊销驾照等刑罚。<br/>2. 民事责任：司机在事故中造成的人身损害或财产损失，可能需要承担民事责任，包括支付医疗费用、赔偿损失等。如果司机没有购买车辆保险，他可能需要承担更多的赔偿责任。<br/>3. 行政责任：根据交通管理法规，驾驶员在饮酒驾车的情况下违反了交通安全规定，可能会被处以行政处罚，例如罚款、吊销驾照、暂扣车辆等。<br/>综上所述，饮酒驾车是一项严重的违法行为，不仅会危及自身安全，还可能对他人造成伤害和损失。如果您或者您的朋友在饮酒后驾车造成事故，应该及时联系当地的律师或法律援助机构，以获得更多的帮助和支持。 |

---

**Input:** 十多个陌生人，深夜砸烂住宅门，撞入并打人，行为人应受到何种处罚，受伤害人如何维护自己的权益

| **Lawyer LLaMA (7B)**                                                                                                                                                                                                                                                                                                   | **ChatGPT (175B)**                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 行为人应该受到刑事处罚和民事赔偿责任。具体的处罚应该根据行为人的具体情况而定，可能包括拘留、罚款、有期徒刑等。同时，行为人还应该承担民事赔偿责任，赔偿受伤害人的医疗费、误工费、精神损失费等相关损失。<br/>受伤害人可以通过以下方式维护自己的权益：<br/>1.报警：受伤害人应该立即报警，向警方提供相关证据，如现场照片、医疗记录等，以便警方进行调查和处理。<br/>2.寻求医疗救治：受伤害人应该及时寻求医疗救治，确保自己的伤势得到妥善治疗。<br/>3.保留证据：受伤害人应该保留现场照片、医疗记录、警方调查报告等相关证据，以便在需要时提供给律师或法院。<br/>4.寻求法律援助：受伤害人可以寻求律师的帮助，了解自己的权益和维权途径。<br/>5.提起民事诉讼：如果行为人拒绝承担民事赔偿责任，受伤害人可以通过民事诉讼的方式维护自己的权益，要求行为人赔偿医疗费、误工费、精神损失费等相关损失。 |这种行为属于非法侵入和暴力行为，违反了法律和社会道德规范。如果这种行为被证实是有预谋的，可能还涉及到犯罪的成分，应该被追究刑事责任。<br/>具体的处罚取决于不同国家或地区的法律体系和司法程序，但通常情况下，这种行为可能会被视为恶性侵入或暴力犯罪，可能会受到拘留、罚款、监禁或者其他刑事处罚。<br/>对于受害人，他们可以采取以下措施来维护自己的权益：<br/>1.立即报警：受害人可以向当地警方报案，说明情况并提供必要的证据，让警方进行调查和追踪行为人。<br/>2.寻求法律援助：如果受害人的人身安全受到威胁或者财产受到损失，他们可以寻求律师的帮助，提起民事诉讼或者申请保护令等法律手段，维护自己的权益。<br/>3.收集证据：受害人可以收集证据，包括现场照片、视频、医院诊断证明等，以证明自己的受害情况和行为人的违法行为，为维护自己的权益提供有力的证据支持。<br/>总之，任何人都应该遵守法律和社会道德规范，不得侵犯他人的人身安全和财产权益。如果发生此类情况，受害人应立即报警并寻求法律援助，以维护自己的权益。 |

</details>

## 致谢

本项目的开放过程中，获得了以下项目的帮助，在此表示感谢。

https://jecqa.thunlp.org/

https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/OLO4G8

https://github.com/tatsu-lab/stanford_alpaca

https://github.com/LianjiaTech/BELLE



本项目贡献人员：

黄曲哲*，陶铭绪*，安震威*，张晨*，姜聪，陈智斌，伍子睿，冯岩松

\* Equal Contribution

本项目是在冯岩松教授的指导下进行的。

## **局限性和使用限制**

本项目内容仅供用于学术研究，不得用于商业以及其他会对社会带来危害的用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。

本项目中使用的数据由ChatGPT生成，未经严格验证，可能会存在错误内容，在使用时请注意甄别。

本项目中的模型输出并非专业法律咨询结果，可能会包含错误内容。如需法律援助，请从专业人士处获得帮助。

## 引用

如果您使用了本项目的内容，或者认为本项目对您的研究有帮助，请引用本项目。

```Plaintext
@misc{Lawyer-LLama,
  author = {Quzhe Huang, Mingxu Tao, Zhenwei An, Chen Zhang, Cong Jiang, Zhibing Chen, Zirui Wu, Yansong Feng},
  title = {Lawyer Llama},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AndrewZhe/lawyer-llama}},
}
```
import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Conv2D, Pool2D
import matplotlib.pyplot as plt

'''
参数配置
'''
# 字典类型
train_parameters = {
    "input_size": [1, 20, 20],  # 输入图片的shape
    "class_dim": -1,  # 分类数
    "src_path": "./characterData.zip",  # 原始数据集路径
    "target_path": "./data/dataset",  # 要解压的路径
    "train_list_path": "./train_data.txt",  # train_data.txt路径
    "eval_list_path": "./val_data.txt",  # eval_data.txt路径
    "label_dict": {},  # 标签字典
    "readme_path": "data/readme.json",  # readme.json路径
    "num_epochs": 1,  # 训练轮数
    "train_batch_size": 32,  # 批次的大小
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.001  # 超参数学习率
    }
}


def unzip_data(src_path, target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至data/dataset目录下
    '''
    # 如果os.path.isdir  path 是 现有的 目录，则返回 True。本方法会跟踪符号链接，因此，对于同一路径，islink() 和 isdir() 都可能为 True。
    if (not os.path.isdir(target_path)):
        # 判断目录中是否有数据集文件
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")


def get_data_list(target_path,train_list_path,eval_list_path):
    '''
    生成数据列表
    '''
    #存放所有类别的信息
    class_detail = []
    #获取所有类别保存的文件夹名称
    data_list_path=target_path
    class_dirs = os.listdir(data_list_path)
    if '__MACOSX' in class_dirs:
        class_dirs.remove('__MACOSX')
    # #总的图像数量 16151
    all_class_images = 0
    # #存放类别标签 65
    class_label=0
    # #存放类别数目 65
    class_dim = 0
    # #存储要写进eval.txt和train.txt中的内容
    trainer_list=[] #训练的14506个文件路径
    eval_list=[]  #测试的1645个文件路径
    #读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            #每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            #统计每个类别有多少张图片
            class_sum = 0
            #获取类别路径
            path = os.path.join(data_list_path,class_dir)
            # print(path)
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:                                  # 遍历文件夹下的每个图片
                if img_path =='.DS_Store':
                    continue
                name_path = os.path.join(path,img_path)                       # 每张图片的路径
                if class_sum % 10 == 0:                                 # 每10张图片取一个做验证数据
                    eval_sum += 1                                       # eval_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")#trainer_sum测试数据的数目
                class_sum += 1                                          #每类图片的数目
                all_class_images += 1                                   #所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir             #类别名称
            class_detail_list['class_label'] = class_label          #类别标签
            class_detail_list['class_eval_images'] = eval_sum       #该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum #该类数据的训练集数目
            class_detail.append(class_detail_list)
            #初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1

    #初始化分类数
    train_parameters['class_dim'] = class_dim
    #乱（测试集）序
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)
    #乱（训练集）序
    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path                 #照片文件父目录
    readjson['all_class_images'] = all_class_images              #总共照片数量16151
    readjson['class_detail'] = class_detail                      #每一类信息
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
#使用这个 转换表 将 obj 序列化为 JSON 格式的 str。当指定时，separators 应当是一个 (item_separator, key_separator) 元组。当 indent 为 None 时，默认值取 (', ', ': ')，否则取 (',', ': ')。为了得到最紧凑的 JSON 表达式，你应该指定其为 (',', ':') 以消除空白字符。
#在 3.4 版更改: 现当 indent 不是 None 时，采用 (',', ': ') 作为默认值。
#当 default 被指定时，其应该是一个函数，每当某个对象无法被序列化时它会被调用。它应该返回该对象的一个可以被 JSON 编码的版本或者引发一个 TypeError。如果没有被指定，则会直接引发 TypeError。
#如果 sort_keys 是 true（默认为 False），那么字典的输出会以键的顺序排序。
    with open(train_parameters['readme_path'],'w') as f:
        f.write(jsons)
    print ('生成数据列表完成！')


## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img



def data_reader(file_list):
    '''
    自定义data_reader
    '''
    def reader():
        with open(file_list, 'r') as f:
            #lines 每一行路径str组合成的list
            lines = [line.strip() for line in f]
            for line in lines:
                #分开 图片路径 标签
                img_path, lab = line.strip().split('\t')
                img = cv_imread(img_path)
                #图片由转换为GRAY 主要看数据是否是uint8
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #图片数据转换为数组并将其类型转换为f32
                img = np.array(img).astype('float32')
                img = img/255.0
                yield img, int(lab)
    return reader

'''
参数初始化
'''
src_path = train_parameters['src_path']
target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']
batch_size = train_parameters['train_batch_size']
'''
解压原始数据到指定路径
'''
unzip_data(src_path, target_path)
# 每次生成数据列表前，首先清空train.txt和eval.txt
# with 简化排错代码
with open(train_list_path, 'w') as f:
    f.seek(0)  # seek() 方法用于移动文件读取指针到指定位置。 此处为开头
    f.truncate()  # truncate() 方法用于截断文件，如果指定了可选参数 size，则表示截断文件为 size 个字符。 如果没有指定 size，则从当前位置起截断；截断之后 size 后面的所有字符被删除。
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

#生成数据列表
get_data_list(target_path,train_list_path,eval_list_path)

'''
构造数据提供器
'''
#该接口是一个reader的装饰器。返回的reader将输入reader的数据打包成指定的batch_size大小的批处理数据（batched data）
train_reader = paddle.batch(data_reader(train_list_path),batch_size=batch_size,drop_last=True)
eval_reader = paddle.batch(data_reader(eval_list_path),batch_size=batch_size,drop_last=True)


Batch=0
Batchs=[]
all_train_accs=[]
def draw_train_acc(Batchs, train_accs):
    title="training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(Batchs, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()

all_train_loss=[]
def draw_train_loss(Batchs, train_loss):
    title="training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs, train_loss, color='red', label='training loss')
    plt.legend()
    plt.grid()
    plt.show()


#表示MyLeNet类继承fluid.dygraph.Layer
class MyLeNet(fluid.dygraph.Layer):
    def __init__(self):
        super(MyLeNet,self).__init__()
        #输入图像的通道数=1  28个卷积核 提取28个特征 大小为5*5 步长1
        self.hidden1_1 = Conv2D(1,28,5,1) #通道数、卷积核个数、卷积核大小
        #池化核大小2×2
        self.hidden1_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=1)
        self.hidden2_1 = Conv2D(28,32,3,1)
        self.hidden2_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=1)
        self.hidden3 = Conv2D(32,32,3,1)
        self.hidden4 = Linear(32*10*10,65,act='softmax')
    def forward(self,input):
        # print(input.shape)
        x = self.hidden1_1(input)
        # print("x_after_h1_1",x.shape)
        x = self.hidden1_2(x)
        # print("x_after_h1_2",x.shape)
        x = self.hidden2_1(x)
        # print("x_after_h2_1",x.shape)
        x = self.hidden2_2(x)
        # print("x_after_h2_2",x.shape)
        x = self.hidden3(x)
        # print("x_after_h3",x.shape)
        x = fluid.layers.reshape(x, shape=[-1, 32 * 10 * 10])
        # print("x_after_reshape", x.shape)
        y = self.hidden4(x)
        # print("x_after_h4", y.shape)
        return y


with fluid.dygraph.guard():
    model = MyLeNet()  # 模型实例化
    model.train()  # 训练模式
    # model.parameters()参数用console打印观看
    opt = fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'],
                                       parameter_list=model.parameters())  # 优化器选用SGD随机梯度下降，学习率为0.001.
    epochs_num = train_parameters['num_epochs']  # 迭代次数=1

    for pass_num in range(epochs_num):
        for batch_id, data in enumerate(train_reader()):
            # 一个批次的照片数据
            images = np.array([x[0].reshape(1, 20, 20) for x in data], np.float32)
            # 一个批次的标签
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]
            image = fluid.dygraph.to_variable(images)
            label = fluid.dygraph.to_variable(labels)

            predict = model(image)  # 数据传入model

            # 该OP实现了softmax交叉熵损失函数。该函数会将softmax操作、交叉熵损失函数的计算过程进行合并，从而提供了数值上更稳定的计算。
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)  # 获取loss值

            acc = fluid.layers.accuracy(predict, label)  # 计算精度

            if batch_id != 0 and batch_id % 50 == 0:
                Batch = Batch + 50
                Batchs.append(Batch)
                # avg_loss转换为ndarray 求其值
                all_train_loss.append(avg_loss.numpy()[0])
                all_train_accs.append(acc.numpy()[0])

                #print(
                #    "train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num, batch_id, avg_loss.numpy(),
                 #                                                                 acc.numpy()))
            # 计算给定的 Tensors 的反向梯度。
            avg_loss.backward()
            opt.minimize(avg_loss)  # 优化器对象的minimize方法对参数进行更新
            model.clear_gradients()  # model.clear_gradients()来重置梯度
    fluid.save_dygraph(model.state_dict(), 'MyLeNet')  # 保存模型

draw_train_acc(Batchs, all_train_accs)
draw_train_loss(Batchs, all_train_loss)

#模型评估
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('MyLeNet')
    model = MyLeNet()
    model.load_dict(model_dict) #加载模型参数
    model.eval() #训练模式
    for batch_id,data in enumerate(eval_reader()):#测试集
        images=np.array([x[0].reshape(1,20,20) for x in data],np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]
        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)
        predict=model(image)
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)


#将标签进行转换
print('Label:',train_parameters['label_dict'])
match = {'A':'A','B':'B','C':'C','D':'D','E':'E','F':'F','G':'G','H':'H','I':'I','J':'J','K':'K','L':'L','M':'M','N':'N',
        'O':'O','P':'P','Q':'Q','R':'R','S':'S','T':'T','U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z',
        'yun':'云','cuan':'川','hei':'黑','zhe':'浙','ning':'宁','jin':'津','gan':'赣','hu':'沪','liao':'辽','jl':'吉','qing':'青','zang':'藏',
        'e1':'鄂','meng':'蒙','gan1':'甘','qiong':'琼','shan':'陕','min':'闽','su':'苏','xin':'新','wan':'皖','jing':'京','xiang':'湘','gui':'贵',
        'yu1':'渝','yu':'豫','ji':'冀','yue':'粤','gui1':'桂','sx':'晋','lu':'鲁',
        '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9'}
L = 0
LABEL ={}
for V in train_parameters['label_dict'].values():
    LABEL[str(L)] = match[V]
    L += 1
print(LABEL)


def load_image(path):
    img = paddle.dataset.image.load_image(file=path, is_color=False)
    img = img.astype('float32')
    img = img[np.newaxis,] / 255.0
    return img


#构建预测动态图过程
with fluid.dygraph.guard():
    model=MyLeNet()#模型实例化
    model_dict,_=fluid.load_dygraph('MyLeNet')
    model.load_dict(model_dict)#加载模型参数
    model.eval()#评估模式
    lab=[]
    for i in range(8):
        if i==2:
            continue
        infer_imgs = []
        infer_imgs.append(load_image('work/' + str(i) + '.png'))
        infer_imgs = np.array(infer_imgs)
        infer_imgs = fluid.dygraph.to_variable(infer_imgs)
        result=model(infer_imgs)
        #找最大数的索引
        lab.append(np.argmax(result.numpy()))
print(lab)
#display(Image.open('work/车牌.png'))
for i in range(len(lab)):
    print(LABEL[str(lab[i])],end='')

a=1
# import torch
# from thop import profile
# from module.GlobalPointer import GlobalPointer  # 从你的模型文件导入模型类
#
# # 1. 创建模型实例（无需训练）
# model = GlobalPointer()
#
# # 2. 准备一个符合输入格式的随机张量（仅用于形状推断）
# dummy_input = torch.randint(0, 200, (4, 256)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     dummy_attention_mask = torch.ones((4, 256), dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     dummy_token_type_ids = torch.randint(0, 2, (4, 256), dtype=torch.long)[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     # 匹配词相关
#     matched_word_ids=torch.randint(0, word_vocab.get_item_size(), (4,256,5)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     matched_word_mask=torch.ones((4, 256, 5), dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     # 边界信息
#     boundary_ids=torch.randint(0, 2, (4, 256), dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 0或1
#     # 标签（仅用于计算图构建，不影响参数量）
#     labels=torch.randint(0, label_vocab.get_item_size(), (4, 5,256,256))[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     # 训练标志（使用字符串）
#     flag="Train"  # 注意这是字符串，不是张量
#     #  radicals相关
#     lattice=torch.randn(4, 256).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 假设每个位置有5个radical
#     seq_len=torch.randint(0, 256, (4,)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     # radicals掩码
#     radical_mask=torch.ones((4, 256), dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 与lattice形状匹配
#
#     # print("--------------------")
#     # print(matched_word_ids)
#
#     flops, params = profile(model, inputs=[dummy_input,dummy_attention_mask,dummy_token_type_ids,matched_word_ids,
#                                            matched_word_mask,boundary_ids,labels,flag,lattice,seq_len,radical_mask])
#     flops, params = clever_format([flops, params], "%.3f")
#     # 4. 输出结果
#     print(f"----------参数量----------: {params}")
#     print(f"----------FLOPs----------: {flops}")
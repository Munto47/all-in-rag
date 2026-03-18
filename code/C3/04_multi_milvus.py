import os
import torch
from tqdm import tqdm
from glob import glob
from visual_bge.visual_bge.modeling import Visualized_BGE
# 注意：混合检索需要用到 AnnSearchRequest 和 RRFRanker
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, AnnSearchRequest, RRFRanker
import numpy as np
import cv2
from PIL import Image

# 1. 配置更新
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_hybrid_demo"
MILVUS_URI = "http://localhost:19530"


# 2. 增强型编码器 (支持多模态融合)
class HybridEncoder:
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_multimodal(self, image_path: str, text: str):
        """融合图像和文本特征生成一个稠密向量"""
        with torch.no_grad():
            vec = self.model.encode(image=image_path, text=text)
        return vec.tolist()[0]

    def encode_image_only(self, image_path: str):
        with torch.no_grad():
            vec = self.model.encode(image=image_path)
        return vec.tolist()[0]

    def generate_sparse_pseudo_vector(self, text: str):
        """
        模拟生成稀疏向量（实际应用中应使用 BGE-M3 或 BM25 算法）。
        这里用词频模拟以演示 Milvus 混合检索流程。
        """
        # 简化演示：将文本哈希映射为稀疏索引
        tokens = text.split()
        return {abs(hash(t)) % 10000: 1.0 for t in tokens}


# 3. 初始化
encoder = HybridEncoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 创建支持混合检索的 Schema
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=768),  # 稠密向量
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),  # 稀疏向量
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=100),  # 标签
]

schema = CollectionSchema(fields, description="多模态混合检索系统")
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 5. 创建双索引
index_params = milvus_client.prepare_index_params()
# 稠密索引 (HNSW)
index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                       params={"M": 16, "efConstruction": 256})
# 稀疏索引 (SPARSE_INVERTED_INDEX)
index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

# 6. 数据入库（包含图像语义 + 文本标签）
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
data_to_insert = []
for path in tqdm(image_list[:50], desc="构建混合索引"):
    # 假设每张图有一个简单的标签（实际可从文件名提取）
    tag_text = "dragon mythical creature"

    dense_vec = encoder.encode_image_only(path)
    sparse_vec = encoder.generate_sparse_pseudo_vector(tag_text)

    data_to_insert.append({
        "dense_vector": dense_vec,
        "sparse_vector": sparse_vec,
        "image_path": path,
        "tag": tag_text
    })

milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
milvus_client.load_collection(COLLECTION_NAME)

# 7. 执行混合检索 (Multi-vector Hybrid Search)
print("\n--> 正在执行多模态融合 + 混合检索...")
query_img = os.path.join(DATA_DIR, "dragon", "query.png")
query_txt = "a golden flying dragon"

# A. 提取检索向量
# 1. 稠密路径：图文融合向量
fused_dense_vec = encoder.encode_multimodal(query_img, query_txt)
# 2. 稀疏路径：文本关键词向量
query_sparse_vec = encoder.generate_sparse_pseudo_vector(query_txt)

# B. 构造混合检索请求
# 路径1: 语义搜索
res_dense = AnnSearchRequest(data=[fused_dense_vec], anns_field="dense_vector", param={"metric_type": "COSINE"},
                             limit=10)
# 路径2: 关键词匹配
res_sparse = AnnSearchRequest(data=[query_sparse_vec], anns_field="sparse_vector", param={"metric_type": "IP"},
                              limit=10)

# C. 使用 RRF 进行重排序融合
results = milvus_client.hybrid_search(
    collection_name=COLLECTION_NAME,
    reqs=[res_dense, res_sparse],
    ranker=RRFRanker(k=60),  # 使用 RRF 算法
    limit=5,
    output_fields=["image_path", "tag"]
)[0]

# 8. 打印与展示
print("\n混合检索结果 (Top 5):")
for hit in results:
    print(f"得分: {hit['distance']:.4f} | 路径: {hit['entity']['image_path']}")

# 资源清理 (可选)
# milvus_client.drop_collection(COLLECTION_NAME)
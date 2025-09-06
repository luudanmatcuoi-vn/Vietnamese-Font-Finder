import torch, types, json
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import chromadb
from chromadb.config import Settings
import numpy as np
import os
import pickle
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

class FontRAGSystem:
    def __init__(self, 
                 embedding_model_path: str = "font_embedding_model.pth",
                 chroma_db_path: str = "./font_database",
                 collection_name: str = "font_embeddings"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load embedding model
        self.embedding_model = self._load_embedding_model(embedding_model_path)
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def _load_embedding_model(self, model_path: str):
        """Load the ResNet50 embedding model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model not found: {model_path}")
        
        print(f"Loading embedding model from: {model_path}")
        model = torch.load(model_path, map_location=self.device, weights_only = False)
        model.to(self.device)
        return model
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def _extract_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract embedding from image tensor"""
        with torch.no_grad():
            embedding = self.embedding_model(image_tensor)
            return embedding.cpu().numpy().flatten()
    
    def _load_font_dict(self, json_path: str):
        try:
            with open(json_path, 'rb') as f:
                font_dict = json.load(f)
            return font_dict
        except Exception as e:
            print(f"Error loading font label {json_path}: {e}")
            return None
    
    def index_images_folder(self, images_folder: str, batch_size: int = 32):        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
        print(f"Found {len(image_files)} images to process")
        
        if not image_files:
            print("No images found in the specified folder")
            return
        
        # Process images in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            self._process_image_batch(batch_files, i)
            
        print(f"Indexing complete! Total documents in collection: {self.collection.count()}")
    
    def _create_item_id(self, obj ):
        otext = obj["text"].replace("\n","\t")
        obj = f'{obj["font"]}||{obj["language"]}||{otext}||{obj["text_size"]}||{obj["text_direction"]}||{obj["angle"]}'
        return obj

    def _process_image_batch(self, image_files: List[str], batch_start_idx: int):    
        embeddings = []
        metadatas = []
        ids = []
        
        for idx, image_path in enumerate(image_files):
            try:
                # Load image and extract embedding
                image_tensor = self._load_image(image_path)
                if image_tensor is None:
                    continue
                
                # Load corresponding font dict
                json_path = os.path.splitext(image_path)[0] + '.json'
        
                if os.path.exists(json_path):
                    font_dict = self._load_font_dict(json_path)
                else:
                    font_dict = {}

                # Prepare metadata
                metadata = font_dict.copy()
                metadata["image_path"] = image_path
                metadata["text_color"] = "_".join([str(c) for c in metadata["text_color"]])
                metadata["bbox"] = "_".join([str(c) for c in metadata["bbox"]])
                if metadata["stroke_color"] == None:
                    metadata["stroke_color"] = "0_0_0"
                else:
                    metadata["stroke_color"] = "_".join([str(c) for c in metadata["stroke_color"]])
                
                # Prepare embedding
                tids = self._create_item_id(metadata)

                check_ids = self.collection.get(ids=[tids], include=[]) 
                if check_ids['ids']:
                    raise ValueError("Image aready in database")

                embedding = self._extract_embedding(image_tensor)

                embeddings.append(embedding.tolist())
                metadatas.append(metadata)
                ids.append( tids )
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {batch_start_idx + idx + 1} images...")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Add to ChromaDB collection
        if embeddings:
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(embeddings)} embeddings to collection")
    
    def search_similar_fonts(self, query_image_path: str, where = {}, n_results: int = 10, unique = False) -> Dict[str, Any]:
        
        # Extract embedding from query image
        query_tensor = self._load_image(query_image_path)
        if query_tensor is None:
            raise ValueError(f"Could not load query image: {query_image_path}")
        
        query_embedding = self._extract_embedding(query_tensor)
        
        # Search in ChromaDB
        if where == {}:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results, self.collection.count()),
                include=['metadatas', 'distances']  )
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results, self.collection.count()),
                where= where,
                include=['metadatas', 'distances']  )
        
        formatted_results = []
        bla = []
        for i in range(len(results['ids'][0])):
            if unique:
                if results['metadatas'][0][i]["font"] in bla:
                    continue
                else:
                    bla.append(results['metadatas'][0][i]["font"])
            result = {
                'rank': i + 1,
                'sim_score': int(( 1 - results['distances'][0][i] )*10000)/10000 ,
            }
            for key in results['metadatas'][0][i].keys():
                result[key] = results['metadatas'][0][i][key]
                        
            formatted_results+=[result]

        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection"""
        total_count = self.collection.count()
        
        # Get sample of metadata to analyze
        if total_count > 0:
            sample_size = min(100, total_count)
            sample_results = self.collection.get(limit=sample_size, include=['metadatas'])
            
            # Analyze font distribution
            font_names = []
            languages = []
            text_sizes = []
            
            for metadata in sample_results['metadatas']:
                if metadata.get('has_font_dict', False):
                    font_names.append(metadata.get('font', 'Unknown'))
                    languages.append(metadata.get('language', 'unknown'))
                    text_sizes.append(metadata.get('text_size', 0))
            
            unique_fonts = len(set(font_names)) if font_names else 0
            unique_languages = len(set(languages)) if languages else 0
            avg_text_size = np.mean(text_sizes) if text_sizes else 0
            
            return {
                'device': self.device,
                'total_images': total_count,
                'images_with_font_dicts': len(font_names),
                'unique_fonts_sampled': unique_fonts,
                'unique_languages_sampled': unique_languages,
                'average_text_size': avg_text_size,
                'sample_size': sample_size,
            }
        
        return {'total_images': 0}
    
    # def search_by_font_name(self, font_name: str, n_results: int = 10) -> Dict[str, Any]:
    #     """Search for images with specific font name"""
        
    #     # Use ChromaDB's where clause to filter by font name
    #     results = self.collection.get(
    #         where={"font_name": {"$eq": font_name}},
    #         limit=n_results,
    #         include=['metadatas', 'documents']
    #     )
        
    #     formatted_results = {
    #         'query_font': font_name,
    #         'total_results': len(results['ids']),
    #         'results': []
    #     }
        
    #     for i in range(len(results['ids'])):
    #         result = {
    #             'rank': i + 1,
    #             'image_path': results['metadatas'][i]['image_path'],
    #             'image_filename': results['metadatas'][i]['image_filename'],
    #             'document': results['documents'][i],
    #             'font_info': {
    #                 'font_name': results['metadatas'][i].get('font_name', 'Unknown'),
    #                 'text': results['metadatas'][i].get('text', ''),
    #                 'text_size': results['metadatas'][i].get('text_size', 0),
    #                 'language': results['metadatas'][i].get('language', 'unknown')
    #             }
    #         }
    #         formatted_results['results'].append(result)
        
    #     return formatted_results
    
    # def export_embeddings(self, output_path: str):
    #     all_data = self.collection.get(
    #         include=['embeddings', 'metadatas']
    #     )

    #     export_data = {
    #         'embeddings': all_data['embeddings'],
    #         'metadatas': all_data['metadatas'],
    #         'ids': all_data['ids'],
    #         'collection_stats': self.get_collection_stats()
    #     }
        
    #     with open(output_path, 'w') as f:
    #         json.dump(export_data, f, indent=2)
    #     print(f"Exported {len(all_data['ids'])} embeddings to {output_path}")
    
    def clear_collection(self):
        self.chroma_client.delete_collection(name=self.collection.name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared")


# if __name__ == "__main__":
#     demo_usage()

# # Additional utility functions
# def batch_search_images(rag_system: FontRAGSystem, query_images: List[str], n_results: int = 5) -> Dict[str, Any]:
    
#     batch_results = {}
    
#     for query_image in query_images:
#         try:
#             results = rag_system.search_similar_fonts(query_image, n_results)
#             batch_results[query_image] = results
#             print(f"Processed query: {os.path.basename(query_image)}")
            
#         except Exception as e:
#             print(f"Error processing {query_image}: {e}")
#             batch_results[query_image] = {'error': str(e)}
    
#     return batch_results

# def create_font_recommendations(search_results: Dict[str, Any], min_similarity: float = 0.7) -> List[Dict[str, Any]]:    
#     recommendations = []
    
#     for result in search_results['results']:
#         if result['similarity_score'] >= min_similarity and 'font_info' in result:
#             recommendation = {
#                 'font_name': result['font_info']['font_name'],
#                 'confidence': result['similarity_score'],
#                 'sample_image': result['image_filename'],
#                 'sample_text': result['font_info']['text'],
#                 'characteristics': {
#                     'text_size': result['font_info']['text_size'],
#                     'language': result['font_info']['language'],
#                     'direction': result['font_info']['text_direction'],
#                     'has_stroke': result['font_info']['stroke_width'] > 0
#                 }
#             }
#             recommendations.append(recommendation)
    
#     # Sort by confidence
#     recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
#     return recommendations
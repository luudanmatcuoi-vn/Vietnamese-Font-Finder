import os
from font_module.rag_system import FontRAGSystem


# Initialize the RAG system
print("Initializing Font RAG System...")
rag_system = FontRAGSystem(
    embedding_model_path="font_embedding_model.pth",
    chroma_db_path="./font_database"
)

# Index images from a folder
images_folder = "./dataset/font_img"  # Change this to your images folder

if os.path.exists(images_folder):

    print(f"\nIndexing images from: {images_folder}")
    rag_system.index_images_folder(images_folder)
    
    # Show collection statistics
    print("\nCollection Statistics:")
    stats = rag_system.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # # Example search by font name
    # print(f"\nSearching for specific font name...")
    # font_search_results = rag_system.search_by_font_name("Arial", n_results=5)
    # print(f"Found {font_search_results['total_results']} images with Arial font")
    
else:
    print(f"Images folder not found: {images_folder}")




# # Example search with a query image
# query_image = "F:\\rank\\font_find\\yuzumaker\\test_font.jpg"  # Change this to your query image

# if os.path.exists(query_image):
#     print(f"\nSearching for similar fonts to: {query_image}")
    
#     search_results = rag_system.search_similar_fonts(
#         query_image_path=query_image,
#         n_results=25
#     )
    
#     print(f"\nFound {len(search_results)} similar font styles:")
    
#     for result in search_results:
#         print(f"\nRank {result['rank']}:")
#         print(f"  Similarity: {result['sim_score']:.4f}")
#         print(f"  Font Name: {result['font_name']}")

from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import CohereEmbeddings
from typing import List, Iterable, Optional, Any

class NewCohereEmbeddings(CohereEmbeddings):
    def embed_documents(self, texts: List[str], input_type: str) -> List[List[float]]:
        embeddings = self.client.embed(
            model=self.model, texts=texts, truncate=self.truncate, input_type=input_type
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    async def aembed_documents(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        embeddings = await self.async_client.embed(
            model=self.model, texts=texts, truncate=self.truncate, input_type=input_type
        )
        return [list(map(float, e)) for e in embeddings.embeddings]


class NewPGVector(PGVector):
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding_function.embed_documents(list(texts), input_type="search_document")
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
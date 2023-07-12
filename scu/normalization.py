import numpy as np

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer


class Normalization:
    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        """Normalization by column

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        norm_matrix = matrix.copy()
        column_sums = norm_matrix.sum(axis=0)
        norm_matrix = norm_matrix.multiply(1 / column_sums)
        norm_matrix = norm_matrix.tocsr()
        return norm_matrix

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        
        return Normalization.by_column(matrix.transpose()).transpose().tocsr()

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """Normalization using tf-idf

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        doc_term_count = matrix.sum(axis=1)
        tf = matrix.multiply(1 / doc_term_count)
        num_documents = matrix.shape[0]
        doc_freq = np.array(matrix.astype(bool).sum(axis=0)).flatten()
        idf = np.log(num_documents / doc_freq)
        matrix_tfidf = csr_matrix(tf.multiply(idf))
        return matrix_tfidf

    @staticmethod
    def bm_25(
        matrix: csr_matrix, k1: float = 2.0, b: float = 0.75
    ) -> csr_matrix:
        """Normalization based on BM-25

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        
        N, M = matrix.shape
        
        doc_term_count = matrix.sum(axis=1)
        doc_term_count_freq = 1 / doc_term_count
        avg_doc_length = np.mean(doc_term_count)
        
        doc_freq = np.array(matrix.astype(bool).sum(axis=0)).flatten()
        idf = np.log(N / doc_freq)
        
        tf1 = matrix.multiply(doc_term_count_freq)
        tf1.data = 1 / tf1.data
        k1_term =  tf1.multiply(1 - b + b * doc_term_count / avg_doc_length)
        k1_term *= k1 / (k1 + 1)
        k1_term.data += 1 / (k1 + 1)
        k1_term = k1_term.power(-1)
        # Multiply term frequency (TF) and inverse document frequency (IDF)
        bm25 = csr_matrix(idf).multiply(k1_term)
        return csr_matrix(bm25)
    

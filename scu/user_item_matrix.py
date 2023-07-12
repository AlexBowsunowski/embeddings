from typing import Dict

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class UserItemMatrix:
    def __init__(self, sales_data: pd.DataFrame):
        """Class initialization. You can make necessary
        calculations here.

        Args:
            sales_data (pd.DataFrame): Sales dataset.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36
            ...

        """
        self._sales_data = sales_data.copy()

        self._user_count = self._sales_data["user_id"].nunique()
        self._item_count = self._sales_data["item_id"].nunique()

        user_idx = np.array(
            list(
                set(
                    self._sales_data['user_id']
                )
            )
        )
        user_idx.sort()
        self._user_map = {
            item: index 
            for index, item in enumerate(user_idx)
        }

        item_idx = np.array(
            list(
                set(
                    self._sales_data['item_id']
                )
            )
        )
        item_idx.sort()
        self._item_map = {
            item: index 
            for index, item in enumerate(item_idx)
        }

        self._matrix  = csr_matrix((self._user_count, self._item_count))

    @property
    def user_count(self) -> int:
        """
        Returns:
            int: the number of users in sales_data.
        """
        return self._user_count

    @property
    def item_count(self) -> int:
        """
        Returns:
            int: the number of items in sales_data.
        """
        return self._item_count

    @property
    def user_map(self) -> Dict[int, int]:
        """Creates a mapping from user_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            user_map (Dict[int, int]):
                {1: 0, 2: 1, 4: 2, 5: 3}

        Returns:
            Dict[int, int]: User map
        """
        return self._user_map

    @property
    def item_map(self) -> Dict[int, int]:
        """Creates a mapping from item_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            item_map (Dict[int, int]):
                {118: 0, 285: 1, 1229: 2, 1688: 3, 2068: 4}

        Returns:
            Dict[int, int]: Item map
        """
        return self._item_map

    @property
    def csr_matrix(self) -> csr_matrix:
        """User items matrix in form of CSR matrix.

        User row_ind, col_ind as
        rows and cols indecies(mapped from user/item map).

        Returns:
            csr_matrix: CSR matrix
        """
        for row in self._sales_data.values:
            user_id, item_id, qty, prices = row
            user_row = self._user_map[user_id]
            item_col = self._item_map[item_id]
            self._matrix[user_row, item_col] = qty
        return self._matrix

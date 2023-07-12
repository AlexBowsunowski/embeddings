WITH k AS 
(
    SELECT *
    FROM default.karpov_express_orders
    WHERE toDate(timestamp) between %(start_date)s AND %(end_date)s

),
user_item_count_sales AS 
(
    SELECT item_id, user_id, SUM(units) as qty 
    FROM k
    GROUP BY user_id, item_id
),
global_item_prices AS 
(
    SELECT item_id, round(avg(price), 2) global_avg_price 
    FROM k
    GROUP BY item_id
)

SELECT 
    u.item_id,
    u.user_id,
    u.qty,
    g.global_avg_price AS price
FROM user_item_count_sales AS u
JOIN global_item_prices AS g 
USING item_id
ORDER BY u.user_id, u.item_id;
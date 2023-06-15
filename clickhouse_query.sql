SELECT d.item_id, user_id, COUNT(*) AS qty, AVG(mean_global_price) AS price
FROM default.karpov_express_orders AS d
JOIN (
    SELECT item_id, ROUND(AVG(price),2) AS mean_global_price
    FROM default.karpov_express_orders
    GROUP BY item_id
) AS im ON d.item_id = im.item_id
WHERE toDate(timestamp) between '2022-10-20' AND '2022-10-23'
GROUP BY d.item_id, user_id
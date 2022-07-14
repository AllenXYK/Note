- SELECT column, another_column, …particle_speed / 2.0 AS half_particle_speed (对结果做了一个除2）
- DISTINCT 用来只保留一个某个特征的数据
- FROM mytable
- WHERE condition(s)  ABS(particle_position) * 10.0 >500
- ORDER BY column ASC/DESC
- LIMIT num_limit OFFSET num_offset;
- LIKE %


| Function	| Description |
| ----------- | ----------- |
| COUNT(*), COUNT(column)	| 计数！COUNT(*) 统计数据行数，COUNT(column) 统计column非NULL的行数 |
| MIN(column)	| 找column最小的一行 |
| MAX(column)	| 找column最大的一行 |
| AVG(column)	| 对column所有行取平均值 |
|SUM(column)	|对column所有行求和|


INNER JOIN 只会保留两个表都存在的数据（还记得之前的交集吗），这看起来意味着一些数据的丢失，在某些场景下会有问题.

真实世界中两个表存在差异很正常，所以我们需要更多的连表方式，也就是本节要介绍的左连接LEFT JOIN,右连接RIGHT JOIN 和 全连接FULL JOIN. 这几个连接方式都会保留不能匹配的行。
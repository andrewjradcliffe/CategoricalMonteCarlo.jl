#
# Date created: 2022-07-11
# Author: aradclif
#
#
############################################################################################
# Some benchmarks of normalization performance

for i = 1:15
    for j = -1:1
        n = (1 << i) + j
        p = rand(n)
        println("normalize1!, n = ", n)
        @btime normalize1!($p)
        println("vnormalize1!, n = ", n)
        @btime vnormalize1!($p)
    end
end

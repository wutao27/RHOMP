function evaluate_high_rhomp(alpha, maxId, Prob, splitVec, filePath, topVal)
    N = size( Prob[1],1 )
    for d = 2:1+length(Prob)
        res = zeros(1 + length(topVal))
        total = 0
        ind = 0
        f = open(filePath)
        for ln in eachline(f)
            ind+=1
            if splitVec[ind]
                continue
            end
            tempArray = readdlm(IOBuffer(ln),Int)
            n = length(tempArray)
            for i = 2:n
                total += 1
                dd = max(2 , min(i-1,d) )
                mats = Prob[dd-1]
                if i == 2
                    traj = [ maxId+1; tempArray[1:i] ]
                else
                    traj = tempArray[i-dd:i]
                end
                tempCol = mats[1][ :,traj[dd] ]
                for i_ = 2:dd
                    tempCol += mats[i_][ :,traj[dd - i_ + 1] ]
                end
                tempCol = tempCol/(1-tempCol[traj[dd],1])
                tempCol[traj[dd],1] = 0
                if tempCol[traj[dd+1],1] <= SMALLNUM
                    res += [1/maxId; zeros(length(topVal)) ]
                    continue
                end
                nzCol = tempCol.nzval
                tempRank = 1
                for tempVal in nzCol
                    if tempVal > tempCol[traj[dd+1],1]
                        tempRank += 1
                    end
                end
                res[1] += 1/tempRank
                for i_ = 1:length(topVal)
                    if tempRank <= topVal[i_]
                        res[1+i_] += 1
                    end
                end
            end
        end
        res = res/total
        println("results for d = $(d): MRR, Precisions")
        println(res)
    end
end


function evaluate_high_mc(maxId, Prob, dicts, splitVec, filePath, topVal)
    for d = 2:1+length(Prob)
        res = zeros(1 + length(topVal))
        total = 0
        ind = 0
        f = open(filePath)
        for ln in eachline(f)
            ind+=1
            if splitVec[ind]
                continue
            end
            tempArray = readdlm(IOBuffer(ln),Int)
            n = length(tempArray)
            for i = 2:n
                total += 1
                dd = max(2 , min(i-1,d) )
                P = Prob[dd-1]
                indDic1 = dicts[dd-1]
                if i == 2
                    traj = [ maxId+1; tempArray[1:i] ]
                else
                    traj = tempArray[i-dd:i]
                end
                key1 = tuple([traj[i_] for i_ = 1:length(traj)-1]...)
                key2 = tuple([traj[i_] for i_ = 2:length(traj)]...)
                if !haskey(indDic1, key1) || !haskey(indDic1, key2)
                    res += [1/maxId; zeros(length(topVal))]
                    continue
                end
                key1 = indDic1[key1]
                key2 = indDic1[key2]
                tempCol = P[:,key1]
                # tempCol = tempCol/(1-tempCol[key2,1])
                # tempCol[key2,1] = 0
                if tempCol[key2,1] == 0
                    res += [1/maxId; zeros(length(topVal)) ]
                    continue
                end
                nzCol = tempCol.nzval
                tempRank = 1
                for tempVal in nzCol
                    if tempVal > tempCol[key2,1]
                        tempRank += 1
                    end
                end
                res[1] += 1/tempRank
                for i_ = 1:length(topVal)
                    if tempRank <= topVal[i_]
                        res[1+i_] += 1
                    end
                end
            end
        end
        res = res/total
        println("results for d = $(d): MRR, Precisions")
        println(res)
    end
end


function evaluate_high_kneser(Prob, dicts, gamas, dist, P_first, gama, splitVec, filePath, topVal)
    N = length(dist)
    for d = 1:1+length(Prob)
        res = zeros(1 + length(topVal))
        total = 0
        ind = 0
        f = open(filePath)
        for ln in eachline(f)
            ind+=1
            if splitVec[ind]
                continue
            end
            tempArray = readdlm(IOBuffer(ln),Int)
            n = length(tempArray)
            for i = 2:n
                total += 1
                dd = min(i-1,d)
                # indDic1 = dicts[dd-1]
                traj = tempArray[i-dd:i]
                if dd > 1
                    indDic1 = dicts[dd-1]
                    key1 = tuple([traj[i_] for i_ = 1:length(traj)-1]...)
                    # key2 = tuple([traj[i_] for i_ = 2:length(traj)]...)
                    while !haskey(indDic1, key1)
                        dd -= 1
                        if dd == 1
                            break
                        end
                        indDic1 = dicts[dd-1]
                        key1 = key1[2:length(key1)]
                        # key2 = key2[2:length(key2)]
                    end
                end
                tempCol = P_first[:, tempArray[i-1]] + gama[tempArray[i-1]] * dist
                tempCol = tempCol/(1 - tempCol[tempArray[i-1],1])
                tempCol[tempArray[i-1],1] = 0
                if dd > 1
                    for i_ = 2:dd
                        indDic1 = dicts[i_-1]
                        key = indDic1[ key1[length(key1)-i_+1:length(key1)] ]
                        tempCol = Prob[i_-1][:, key ] + gamas[i_-1][key] * tempCol
                    end
                end
                tempRank = 1
                for j_ = 1:length(tempCol)
                    if tempCol[j_,1] > tempCol[tempArray[i],1]
                        tempRank+=1
                    end
                end
                res[1] += 1/tempRank
                for i_ = 1:length(topVal)
                    if tempRank <= topVal[i_]
                        res[1+i_] += 1
                    end
                end
            end
        end
        res = res/total
        println("results for d = $(d): MRR, Precisions")
        println(res)
    end
end
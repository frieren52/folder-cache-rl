using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

public static class FolderLruAnalysis
{
    const ulong BandwidthBytesPerSecond = 200UL * 1024UL * 1024UL;
    const byte HitCode = 255;
    static readonly int[] CapacityPercents = new int[] { 1, 3, 10 };
    static readonly int[] ChannelCounts = new int[] { 10, 30, 60 };
    static readonly string[] CauseCodes = new string[] {
        "cold_start",
        "capacity_eviction",
        "oversize",
        "bandwidth_queue",
        "bandwidth_transfer",
        "bandwidth_state_divergence"
    };
    static readonly string[] OperationalCodes = new string[] {
        "trigger_new_transfer",
        "queued_merge",
        "inflight_merge"
    };
    static readonly string[] WaitBucketCodes = new string[] {
        "lt_1s",
        "1_5s",
        "5_30s",
        "30_60s",
        "1_5m",
        "5_30m",
        "30_60m",
        "ge_1h"
    };

    sealed class Dataset
    {
        public int ObjectCount;
        public int AccessedObjectCount;
        public int EventCount;
        public int LogFileCount;
        public ulong TotalObjectBytes;
        public ulong TotalRequestBytes;
        public ulong FirstAccessBytes;
        public ulong[] Sizes;
        public int[] ExpectedAccessCounts;
        public string[] Paths;
        public int[] FileIds;
        public int[] Times;
        public DateTime StartTime;
        public DateTime EndTime;
    }

    sealed class Result
    {
        public string Scenario;
        public string Mode;
        public int Channels;
        public int CapacityPercent;
        public ulong CapacityBytes;
        public long HitEvents;
        public ulong HitBytes;
        public long SubmittedTransfers;
        public long CompletedTransfersAtLogEnd;
        public int QueuedAtLogEnd;
        public int InflightAtLogEnd;
        public int MaxQueued;
        public ulong FinalCacheUsedBytes;
        public int FinalCacheObjectCount;
        public long Evictions;
        public ulong EvictedBytes;
        public long RefetchTransfers;
        public ulong RefetchBytes;
        public Attribution Attribution;
    }

    sealed class Attribution
    {
        public readonly long[] CauseEvents = new long[CauseCodes.Length];
        public readonly ulong[] CauseBytes = new ulong[CauseCodes.Length];
        public readonly long[] OperationalEvents = new long[OperationalCodes.Length];
        public readonly ulong[] OperationalBytes = new ulong[OperationalCodes.Length];
        public readonly long[] WaitBucketEvents = new long[WaitBucketCodes.Length];
        public readonly ulong[] WaitBucketBytes = new ulong[WaitBucketCodes.Length];
        public readonly long[] ObjectMissEvents;
        public readonly long[][] ObjectCauseEvents;
        public double TotalWaitSeconds;
        public double MaxWaitSeconds;
        public long CounterfactualFiniteOnlyHits;
        public ulong CounterfactualFiniteOnlyHitBytes;

        public Attribution(int objectCount)
        {
            ObjectMissEvents = new long[objectCount];
            ObjectCauseEvents = new long[CauseCodes.Length][];
            for (int i = 0; i < CauseCodes.Length; i++)
                ObjectCauseEvents[i] = new long[objectCount];
        }

        public void AddCause(int cause, int id, ulong bytes)
        {
            CauseEvents[cause]++;
            CauseBytes[cause] = checked(CauseBytes[cause] + bytes);
            ObjectMissEvents[id]++;
            ObjectCauseEvents[cause][id]++;
        }

        public void AddOperational(int state, ulong bytes)
        {
            OperationalEvents[state]++;
            OperationalBytes[state] = checked(OperationalBytes[state] + bytes);
        }

        public void AddWait(double waitSeconds, ulong bytes)
        {
            if (waitSeconds < -1e-9) throw new InvalidOperationException("请求等待时间为负");
            double wait = Math.Max(0.0, waitSeconds);
            TotalWaitSeconds += wait;
            if (wait > MaxWaitSeconds) MaxWaitSeconds = wait;
            int bucket;
            if (wait < 1.0) bucket = 0;
            else if (wait < 5.0) bucket = 1;
            else if (wait < 30.0) bucket = 2;
            else if (wait < 60.0) bucket = 3;
            else if (wait < 300.0) bucket = 4;
            else if (wait < 1800.0) bucket = 5;
            else if (wait < 3600.0) bucket = 6;
            else bucket = 7;
            WaitBucketEvents[bucket]++;
            WaitBucketBytes[bucket] = checked(WaitBucketBytes[bucket] + bytes);
        }
    }

    sealed class FileLru
    {
        readonly ulong capacity;
        readonly ulong[] sizes;
        readonly int[] previous;
        readonly int[] next;
        readonly byte[] state;
        int head;
        int tail;
        int objectCount;
        ulong used;
        long evictions;
        ulong evictedBytes;

        public FileLru(int fileCount, ulong capacityBytes, ulong[] fileSizes)
        {
            capacity = capacityBytes;
            sizes = fileSizes;
            previous = new int[fileCount];
            next = new int[fileCount];
            state = new byte[fileCount];
        }

        public byte State(int id) { return state[id]; }
        public ulong UsedBytes { get { return used; } }
        public int ObjectCount { get { return objectCount; } }
        public long Evictions { get { return evictions; } }
        public ulong EvictedBytes { get { return evictedBytes; } }

        public void MarkQueued(int id)
        {
            if (state[id] != 0) throw new InvalidOperationException("只有缺失对象才能进入队列");
            state[id] = 2;
        }

        public void MarkInflight(int id)
        {
            if (state[id] != 0 && state[id] != 2)
                throw new InvalidOperationException("传输对象状态异常");
            state[id] = 3;
        }

        public void Touch(int id)
        {
            if (state[id] != 1) throw new InvalidOperationException("Touch 非缓存对象");
            int link = id + 1;
            if (head == link) return;
            RemoveLink(id);
            AddHead(id);
        }

        public void InsertInstant(int id)
        {
            if (state[id] != 0) throw new InvalidOperationException("瞬时插入对象状态异常");
            Insert(id);
        }

        public void CompleteTransfer(int id)
        {
            if (state[id] != 3) throw new InvalidOperationException("完成传输对象状态异常");
            Insert(id);
        }

        void Insert(int id)
        {
            ulong size = sizes[id];
            if (size > capacity)
            {
                state[id] = 0;
                return;
            }
            while (tail != 0 && used + size > capacity)
            {
                int victim = tail - 1;
                RemoveLink(victim);
                used -= sizes[victim];
                evictions++;
                evictedBytes = checked(evictedBytes + sizes[victim]);
                state[victim] = 0;
                objectCount--;
            }
            state[id] = 1;
            AddHead(id);
            used += size;
            objectCount++;
            if (used > capacity) throw new InvalidOperationException("缓存容量溢出");
        }

        void RemoveLink(int id)
        {
            int prevLink = previous[id];
            int nextLink = next[id];
            if (prevLink == 0) head = nextLink;
            else next[prevLink - 1] = nextLink;
            if (nextLink == 0) tail = prevLink;
            else previous[nextLink - 1] = prevLink;
            previous[id] = 0;
            next[id] = 0;
        }

        void AddHead(int id)
        {
            int link = id + 1;
            previous[id] = 0;
            next[id] = head;
            if (head != 0) previous[head - 1] = link;
            head = link;
            if (tail == 0) tail = link;
        }
    }

    sealed class LimitedSimulator
    {
        readonly FileLru cache;
        readonly ulong[] sizes;
        readonly Attribution attribution;
        readonly int channelCount;
        readonly int[] queueIds;
        int queueHead;
        int queueTail;
        int queueCount;

        readonly int[] heapFiles;
        readonly double[] heapFinishes;
        readonly long[] heapSequences;
        int heapCount;
        long nextSequence;

        readonly double[] scheduledAvailability;
        int scheduledCount;
        readonly double[] readyTimes;
        readonly bool[] everSubmitted;

        public long HitEvents;
        public ulong HitBytes;
        public long MissEvents;
        public long SubmittedTransfers;
        public long CompletedTransfers;
        public int MaxQueued;

        public LimitedSimulator(
            int fileCount,
            ulong capacity,
            ulong[] fileSizes,
            int channels,
            Attribution missAttribution
        )
        {
            cache = new FileLru(fileCount, capacity, fileSizes);
            sizes = fileSizes;
            attribution = missAttribution;
            channelCount = channels;
            queueIds = new int[fileCount];
            heapFiles = new int[channels];
            heapFinishes = new double[channels];
            heapSequences = new long[channels];
            scheduledAvailability = new double[channels];
            readyTimes = new double[fileCount];
            everSubmitted = new bool[fileCount];
        }

        public int Queued { get { return queueCount; } }
        public int Inflight { get { return heapCount; } }
        public ulong CacheUsedBytes { get { return cache.UsedBytes; } }
        public int CacheObjectCount { get { return cache.ObjectCount; } }
        public long Evictions { get { return cache.Evictions; } }
        public ulong EvictedBytes { get { return cache.EvictedBytes; } }
        public long RefetchTransfers { get; private set; }
        public ulong RefetchBytes { get; private set; }

        public void Access(int id, double timestamp, byte instantCause)
        {
            Advance(timestamp);
            byte state = cache.State(id);
            if (state == 1)
            {
                HitEvents++;
                HitBytes += sizes[id];
                cache.Touch(id);
                if (instantCause != HitCode)
                {
                    attribution.CounterfactualFiniteOnlyHits++;
                    attribution.CounterfactualFiniteOnlyHitBytes = checked(
                        attribution.CounterfactualFiniteOnlyHitBytes + sizes[id]
                    );
                }
                return;
            }

            MissEvents++;
            int cause;
            if (instantCause != HitCode)
            {
                if (instantCause > 2) throw new InvalidDataException("瞬时原因代码异常");
                cause = instantCause;
            }
            else if (state == 2) cause = 3;
            else if (state == 3) cause = 4;
            else if (state == 0) cause = 5;
            else throw new InvalidOperationException("有限带宽 MISS 状态异常");
            attribution.AddCause(cause, id, sizes[id]);

            int operationalState;
            if (state == 0)
            {
                operationalState = 0;
                SubmitTransfer(id, timestamp);
            }
            else if (state == 2) operationalState = 1;
            else if (state == 3) operationalState = 2;
            else throw new InvalidOperationException("有限带宽操作状态异常");
            attribution.AddOperational(operationalState, sizes[id]);
            attribution.AddWait(readyTimes[id] - timestamp, sizes[id]);
        }

        void SubmitTransfer(int id, double timestamp)
        {
            if (everSubmitted[id])
            {
                RefetchTransfers++;
                RefetchBytes = checked(RefetchBytes + sizes[id]);
            }
            everSubmitted[id] = true;
            SubmittedTransfers++;
            readyTimes[id] = ScheduleReady(timestamp, sizes[id]);
            if (heapCount < channelCount)
                StartTransfer(id, timestamp);
            else
                Enqueue(id);
        }

        double ScheduleReady(double timestamp, ulong size)
        {
            double start;
            if (scheduledCount < channelCount)
            {
                start = timestamp;
            }
            else
            {
                start = scheduledAvailability[0];
                scheduledCount--;
                if (scheduledCount > 0)
                {
                    double moved = scheduledAvailability[scheduledCount];
                    int index = 0;
                    while (true)
                    {
                        int left = index * 2 + 1;
                        if (left >= scheduledCount) break;
                        int right = left + 1;
                        int child = right < scheduledCount && scheduledAvailability[right] < scheduledAvailability[left]
                            ? right
                            : left;
                        if (moved <= scheduledAvailability[child]) break;
                        scheduledAvailability[index] = scheduledAvailability[child];
                        index = child;
                    }
                    scheduledAvailability[index] = moved;
                }
                if (start < timestamp) start = timestamp;
            }

            double finish = start + (double)size / (double)BandwidthBytesPerSecond;
            int insert = scheduledCount++;
            while (insert > 0)
            {
                int parent = (insert - 1) >> 1;
                if (finish >= scheduledAvailability[parent]) break;
                scheduledAvailability[insert] = scheduledAvailability[parent];
                insert = parent;
            }
            scheduledAvailability[insert] = finish;
            return finish;
        }

        void Advance(double timestamp)
        {
            while (heapCount > 0 && heapFinishes[0] <= timestamp + 1e-12)
            {
                int completedFile;
                double completedAt;
                PopCompletion(out completedFile, out completedAt);
                cache.CompleteTransfer(completedFile);
                CompletedTransfers++;

                if (queueCount > 0)
                {
                    int nextId = Dequeue();
                    StartTransfer(nextId, completedAt);
                }
            }
        }

        void StartTransfer(int id, double start)
        {
            cache.MarkInflight(id);
            double finish = start + (double)sizes[id] / (double)BandwidthBytesPerSecond;
            PushCompletion(id, finish, nextSequence++);
        }

        void Enqueue(int id)
        {
            if (queueCount >= queueIds.Length)
                throw new InvalidOperationException("等待队列超过对象数");
            cache.MarkQueued(id);
            queueIds[queueTail] = id;
            queueTail++;
            if (queueTail == queueIds.Length) queueTail = 0;
            queueCount++;
            if (queueCount > MaxQueued) MaxQueued = queueCount;
        }

        int Dequeue()
        {
            int id = queueIds[queueHead];
            queueHead++;
            if (queueHead == queueIds.Length) queueHead = 0;
            queueCount--;
            return id;
        }

        bool Less(double finish, long sequence, int index)
        {
            return finish < heapFinishes[index] ||
                (finish == heapFinishes[index] && sequence < heapSequences[index]);
        }

        void PushCompletion(int id, double finish, long sequence)
        {
            int index = heapCount++;
            while (index > 0)
            {
                int parent = (index - 1) >> 1;
                if (!Less(finish, sequence, parent)) break;
                heapFiles[index] = heapFiles[parent];
                heapFinishes[index] = heapFinishes[parent];
                heapSequences[index] = heapSequences[parent];
                index = parent;
            }
            heapFiles[index] = id;
            heapFinishes[index] = finish;
            heapSequences[index] = sequence;
        }

        void PopCompletion(out int id, out double finish)
        {
            id = heapFiles[0];
            finish = heapFinishes[0];
            heapCount--;
            if (heapCount == 0) return;

            int movedId = heapFiles[heapCount];
            double movedFinish = heapFinishes[heapCount];
            long movedSequence = heapSequences[heapCount];
            int index = 0;
            while (true)
            {
                int left = index * 2 + 1;
                if (left >= heapCount) break;
                int right = left + 1;
                int child = left;
                if (right < heapCount &&
                    (heapFinishes[right] < heapFinishes[left] ||
                     (heapFinishes[right] == heapFinishes[left] && heapSequences[right] < heapSequences[left])))
                    child = right;
                if (movedFinish < heapFinishes[child] ||
                    (movedFinish == heapFinishes[child] && movedSequence <= heapSequences[child]))
                    break;
                heapFiles[index] = heapFiles[child];
                heapFinishes[index] = heapFinishes[child];
                heapSequences[index] = heapSequences[child];
                index = child;
            }
            heapFiles[index] = movedId;
            heapFinishes[index] = movedFinish;
            heapSequences[index] = movedSequence;
        }
    }

    static int ParseDigits(string text, int start, int length)
    {
        int value = 0;
        for (int i = 0; i < length; i++)
        {
            char c = text[start + i];
            if (c < '0' || c > '9') throw new InvalidDataException("数字字段异常");
            value = checked(value * 10 + (c - '0'));
        }
        return value;
    }

    static ulong ParseUnsigned(string text, int start, int length)
    {
        ulong value = 0;
        for (int i = 0; i < length; i++)
        {
            char c = text[start + i];
            if (c < '0' || c > '9') throw new InvalidDataException("无符号整数字段异常");
            value = checked(value * 10UL + (ulong)(c - '0'));
        }
        return value;
    }

    static Dataset ReadCatalog(string root)
    {
        string path = Path.Combine(root, "02_全局数据", "artifacts", "vocab", "path_catalog.csv");
        List<ulong> sizes = new List<ulong>();
        List<int> accessCounts = new List<int>();
        List<string> objectPaths = new List<string>();
        ulong totalObjectBytes = 0;
        ulong totalRequestBytes = 0;
        ulong firstAccessBytes = 0;
        long eventCount = 0;
        int accessedObjects = 0;

        using (StreamReader reader = new StreamReader(path, Encoding.UTF8, true, 1024 * 1024))
        {
            string header = reader.ReadLine();
            if (header == null || !header.Contains("path_index") || !header.Contains("total_size_bytes"))
                throw new InvalidDataException("path_catalog.csv 表头异常");
            string line;
            int row = 0;
            while ((line = reader.ReadLine()) != null)
            {
                int firstSep = line.IndexOf("\",\"", StringComparison.Ordinal);
                int lastSep = line.LastIndexOf("\",\"", StringComparison.Ordinal);
                int secondLastSep = lastSep > 0
                    ? line.LastIndexOf("\",\"", lastSep - 1, StringComparison.Ordinal)
                    : -1;
                if (line.Length < 7 || line[0] != '"' || line[line.Length - 1] != '"' ||
                    firstSep < 0 || secondLastSep <= firstSep || lastSep <= secondLastSep)
                    throw new InvalidDataException("path_catalog.csv 行格式异常：" + (row + 2));

                int id = ParseDigits(line, 1, firstSep - 1);
                int pathStart = firstSep + 3;
                int accessStart = secondLastSep + 3;
                int sizeStart = lastSep + 3;
                string objectPath = line.Substring(pathStart, secondLastSep - pathStart);
                int accesses = ParseDigits(line, accessStart, lastSep - accessStart);
                ulong size = ParseUnsigned(line, sizeStart, line.Length - 1 - sizeStart);
                if (id != row) throw new InvalidDataException("path_index 不连续：" + id);

                sizes.Add(size);
                accessCounts.Add(accesses);
                objectPaths.Add(objectPath.Replace("\"\"", "\""));
                totalObjectBytes = checked(totalObjectBytes + size);
                eventCount = checked(eventCount + accesses);
                totalRequestBytes = checked(totalRequestBytes + size * (ulong)accesses);
                if (accesses > 0)
                {
                    accessedObjects++;
                    firstAccessBytes = checked(firstAccessBytes + size);
                }
                row++;
            }
        }

        if (sizes.Count == 0 || eventCount <= 0 || eventCount > int.MaxValue)
            throw new InvalidDataException("对象表规模异常");

        return new Dataset {
            ObjectCount = sizes.Count,
            AccessedObjectCount = accessedObjects,
            EventCount = (int)eventCount,
            TotalObjectBytes = totalObjectBytes,
            TotalRequestBytes = totalRequestBytes,
            FirstAccessBytes = firstAccessBytes,
            Sizes = sizes.ToArray(),
            ExpectedAccessCounts = accessCounts.ToArray(),
            Paths = objectPaths.ToArray()
        };
    }

    static void ParseLogLine(
        string line,
        string source,
        long lineNumber,
        ref int cachedDate,
        ref long cachedDaySeconds,
        out int id,
        out long timestamp
    )
    {
        int p1 = line.IndexOf(' ');
        int p2 = p1 < 0 ? -1 : line.IndexOf(' ', p1 + 1);
        if (p1 <= 0 || p2 - p1 != 11 || line.Length - p2 != 9 ||
            line.IndexOf(' ', p2 + 1) >= 0)
            throw new InvalidDataException("日志格式异常：" + source + ":" + lineNumber);

        int dateStart = p1 + 1;
        int timeStart = p2 + 1;
        if (line[dateStart + 4] != '-' || line[dateStart + 7] != '-' ||
            line[timeStart + 2] != ':' || line[timeStart + 5] != ':')
            throw new InvalidDataException("日志日期格式异常：" + source + ":" + lineNumber);

        id = ParseDigits(line, 0, p1);
        int year = ParseDigits(line, dateStart, 4);
        int month = ParseDigits(line, dateStart + 5, 2);
        int day = ParseDigits(line, dateStart + 8, 2);
        int dateKey = year * 10000 + month * 100 + day;
        if (dateKey != cachedDate)
        {
            cachedDate = dateKey;
            cachedDaySeconds = new DateTime(year, month, day).Ticks / TimeSpan.TicksPerSecond;
        }
        int hour = ParseDigits(line, timeStart, 2);
        int minute = ParseDigits(line, timeStart + 3, 2);
        int second = ParseDigits(line, timeStart + 6, 2);
        timestamp = cachedDaySeconds + hour * 3600L + minute * 60L + second;
    }

    static void ReadEvents(string root, Dataset data)
    {
        string rawDirectory = Path.Combine(root, "02_全局数据", "raw");
        string[] paths = Directory.GetFiles(rawDirectory, "access_*.txt");
        Array.Sort(paths, StringComparer.Ordinal);
        if (paths.Length != 7)
            throw new InvalidDataException("应有 7 份 access 日志，实际为 " + paths.Length);

        int[] ids = new int[data.EventCount];
        int[] times = new int[data.EventCount];
        int[] actualAccessCounts = new int[data.ObjectCount];
        int eventIndex = 0;
        long firstTimestamp = long.MinValue;
        long previousTimestamp = long.MinValue;
        long lastTimestamp = long.MinValue;
        ulong actualRequestBytes = 0;
        DateTime started = DateTime.UtcNow;

        foreach (string path in paths)
        {
            Console.WriteLine("reading " + Path.GetFileName(path));
            int cachedDate = -1;
            long cachedDaySeconds = 0;
            using (StreamReader reader = new StreamReader(
                path,
                new UTF8Encoding(false, true),
                false,
                8 * 1024 * 1024
            ))
            {
                string line;
                long lineNumber = 0;
                while ((line = reader.ReadLine()) != null)
                {
                    lineNumber++;
                    int id;
                    long timestamp;
                    ParseLogLine(
                        line,
                        Path.GetFileName(path),
                        lineNumber,
                        ref cachedDate,
                        ref cachedDaySeconds,
                        out id,
                        out timestamp
                    );
                    if (id < 0 || id >= data.ObjectCount)
                        throw new InvalidDataException("对象编号越界：" + id);
                    if (timestamp < previousTimestamp)
                        throw new InvalidDataException("日志全局时间逆序：" + Path.GetFileName(path) + ":" + lineNumber);
                    if (eventIndex >= data.EventCount)
                        throw new InvalidDataException("日志事件数超过对象表汇总值");
                    if (firstTimestamp == long.MinValue) firstTimestamp = timestamp;
                    long offset = timestamp - firstTimestamp;
                    if (offset < 0 || offset > int.MaxValue)
                        throw new InvalidDataException("日志时间跨度异常");

                    ids[eventIndex] = id;
                    times[eventIndex] = (int)offset;
                    actualAccessCounts[id]++;
                    actualRequestBytes = checked(actualRequestBytes + data.Sizes[id]);
                    eventIndex++;
                    previousTimestamp = timestamp;
                    lastTimestamp = timestamp;
                    if (eventIndex % 5000000 == 0)
                        Console.WriteLine(
                            "events {0:N0}/{1:N0}, {2:F1}s",
                            eventIndex,
                            data.EventCount,
                            (DateTime.UtcNow - started).TotalSeconds
                        );
                }
            }
        }

        if (eventIndex != data.EventCount)
            throw new InvalidDataException("日志事件数不一致：" + eventIndex + " != " + data.EventCount);
        if (actualRequestBytes != data.TotalRequestBytes)
            throw new InvalidDataException("按对象大小加权的请求字节总数不一致");
        for (int i = 0; i < actualAccessCounts.Length; i++)
            if (actualAccessCounts[i] != data.ExpectedAccessCounts[i])
                throw new InvalidDataException("对象访问次数与 path_catalog 不一致：" + i);

        data.FileIds = ids;
        data.Times = times;
        data.LogFileCount = paths.Length;
        data.StartTime = new DateTime(firstTimestamp * TimeSpan.TicksPerSecond);
        data.EndTime = new DateTime(lastTimestamp * TimeSpan.TicksPerSecond);
    }

    static Result SimulateInstant(
        Dataset data,
        int capacityPercent,
        ulong capacity,
        out byte[] causes
    )
    {
        FileLru cache = new FileLru(data.ObjectCount, capacity, data.Sizes);
        Attribution attribution = new Attribution(data.ObjectCount);
        bool[] seen = new bool[data.ObjectCount];
        causes = new byte[data.EventCount];
        long hitEvents = 0;
        ulong hitBytes = 0;
        DateTime started = DateTime.UtcNow;
        for (int i = 0; i < data.EventCount; i++)
        {
            int id = data.FileIds[i];
            if (cache.State(id) == 1)
            {
                hitEvents++;
                hitBytes += data.Sizes[id];
                cache.Touch(id);
                causes[i] = HitCode;
            }
            else
            {
                int cause = data.Sizes[id] > capacity ? 2 : (!seen[id] ? 0 : 1);
                causes[i] = (byte)cause;
                attribution.AddCause(cause, id, data.Sizes[id]);
                cache.InsertInstant(id);
            }
            seen[id] = true;
            if ((i + 1) % 10000000 == 0)
                Console.WriteLine(
                    "instant {0}%: {1:N0}/{2:N0}, {3:F1}s",
                    capacityPercent,
                    i + 1,
                    data.EventCount,
                    (DateTime.UtcNow - started).TotalSeconds
                );
        }
        return new Result {
            Scenario = "瞬时取回",
            Mode = "instant",
            Channels = 0,
            CapacityPercent = capacityPercent,
            CapacityBytes = capacity,
            HitEvents = hitEvents,
            HitBytes = hitBytes,
            FinalCacheUsedBytes = cache.UsedBytes,
            FinalCacheObjectCount = cache.ObjectCount,
            Evictions = cache.Evictions,
            EvictedBytes = cache.EvictedBytes,
            Attribution = attribution
        };
    }

    static Result SimulateLimited(
        Dataset data,
        int channels,
        int capacityPercent,
        ulong capacity,
        byte[] instantCauses
    )
    {
        if (instantCauses == null || instantCauses.Length != data.EventCount)
            throw new InvalidDataException("瞬时反事实原因数组长度异常");
        Attribution attribution = new Attribution(data.ObjectCount);
        LimitedSimulator simulator = new LimitedSimulator(
            data.ObjectCount,
            capacity,
            data.Sizes,
            channels,
            attribution
        );
        DateTime started = DateTime.UtcNow;
        for (int i = 0; i < data.EventCount; i++)
        {
            simulator.Access(data.FileIds[i], data.Times[i], instantCauses[i]);
            if ((i + 1) % 10000000 == 0)
                Console.WriteLine(
                    "limited {0}ch {1}%: {2:N0}/{3:N0}, queue={4:N0}, {5:F1}s",
                    channels,
                    capacityPercent,
                    i + 1,
                    data.EventCount,
                    simulator.Queued,
                    (DateTime.UtcNow - started).TotalSeconds
                );
        }
        return new Result {
            Scenario = channels + "通道 × 200 MiB/s",
            Mode = "finite",
            Channels = channels,
            CapacityPercent = capacityPercent,
            CapacityBytes = capacity,
            HitEvents = simulator.HitEvents,
            HitBytes = simulator.HitBytes,
            SubmittedTransfers = simulator.SubmittedTransfers,
            CompletedTransfersAtLogEnd = simulator.CompletedTransfers,
            QueuedAtLogEnd = simulator.Queued,
            InflightAtLogEnd = simulator.Inflight,
            MaxQueued = simulator.MaxQueued,
            FinalCacheUsedBytes = simulator.CacheUsedBytes,
            FinalCacheObjectCount = simulator.CacheObjectCount,
            Evictions = simulator.Evictions,
            EvictedBytes = simulator.EvictedBytes,
            RefetchTransfers = simulator.RefetchTransfers,
            RefetchBytes = simulator.RefetchBytes,
            Attribution = attribution
        };
    }

    static void ValidateResult(Dataset data, Result result)
    {
        long misses = data.EventCount - result.HitEvents;
        ulong missBytes = data.TotalRequestBytes - result.HitBytes;
        if (result.HitEvents < 0 || result.HitEvents > data.EventCount)
            throw new InvalidDataException("命中事件数不闭合：" + result.Scenario);
        if (result.HitBytes > data.TotalRequestBytes)
            throw new InvalidDataException("命中字节数超过请求字节数：" + result.Scenario);
        if (result.FinalCacheUsedBytes > result.CapacityBytes)
            throw new InvalidDataException("缓存占用超过容量：" + result.Scenario);
        if (result.Attribution == null)
            throw new InvalidDataException("缺少 MISS 归因：" + result.Scenario);

        long causeEvents = 0;
        ulong causeBytes = 0;
        long objectMissEvents = 0;
        for (int i = 0; i < CauseCodes.Length; i++)
        {
            causeEvents = checked(causeEvents + result.Attribution.CauseEvents[i]);
            causeBytes = checked(causeBytes + result.Attribution.CauseBytes[i]);
            long objectCauseEvents = 0;
            for (int id = 0; id < data.ObjectCount; id++)
                objectCauseEvents = checked(objectCauseEvents + result.Attribution.ObjectCauseEvents[i][id]);
            if (objectCauseEvents != result.Attribution.CauseEvents[i])
                throw new InvalidDataException("逐对象原因不闭合：" + result.Scenario + "/" + CauseCodes[i]);
        }
        for (int id = 0; id < data.ObjectCount; id++)
            objectMissEvents = checked(objectMissEvents + result.Attribution.ObjectMissEvents[id]);
        if (causeEvents != misses || objectMissEvents != misses)
            throw new InvalidDataException("MISS 原因次数不闭合：" + result.Scenario);
        if (causeBytes != missBytes)
            throw new InvalidDataException("MISS 原因字节不闭合：" + result.Scenario);

        if (result.Mode == "finite")
        {
            if (result.SubmittedTransfers > misses)
                throw new InvalidDataException("物理传输数超过 miss 数：" + result.Scenario);
            if (result.RefetchTransfers > result.SubmittedTransfers)
                throw new InvalidDataException("重复取回数超过物理传输数：" + result.Scenario);
            if (result.CompletedTransfersAtLogEnd + result.QueuedAtLogEnd + result.InflightAtLogEnd != result.SubmittedTransfers)
                throw new InvalidDataException("传输任务数不闭合：" + result.Scenario);

            long operationalEvents = 0;
            ulong operationalBytes = 0;
            for (int i = 0; i < OperationalCodes.Length; i++)
            {
                operationalEvents = checked(operationalEvents + result.Attribution.OperationalEvents[i]);
                operationalBytes = checked(operationalBytes + result.Attribution.OperationalBytes[i]);
            }
            if (operationalEvents != misses || operationalBytes != missBytes)
                throw new InvalidDataException("有限带宽操作状态不闭合：" + result.Scenario);

            long waitEvents = 0;
            ulong waitBytes = 0;
            for (int i = 0; i < WaitBucketCodes.Length; i++)
            {
                waitEvents = checked(waitEvents + result.Attribution.WaitBucketEvents[i]);
                waitBytes = checked(waitBytes + result.Attribution.WaitBucketBytes[i]);
            }
            if (waitEvents != misses || waitBytes != missBytes)
                throw new InvalidDataException("等待时间桶不闭合：" + result.Scenario);
        }
        else
        {
            for (int i = 3; i < CauseCodes.Length; i++)
                if (result.Attribution.CauseEvents[i] != 0 || result.Attribution.CauseBytes[i] != 0)
                    throw new InvalidDataException("瞬时场景出现带宽原因：" + result.Scenario);
        }
    }

    static string Csv(string value)
    {
        if (value == null) return "";
        if (value.IndexOfAny(new char[] { ',', '\"', '\r', '\n' }) < 0) return value;
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    static void WriteOutputs(string root, Dataset data, List<Result> results)
    {
        string tables = Path.Combine(root, "02_全局数据", "analysis", "lru_baseline_analysis", "tables");
        string attributionTables = Path.Combine(root, "02_全局数据", "analysis", "lru_miss_attribution_analysis", "tables");
        Directory.CreateDirectory(tables);
        Directory.CreateDirectory(attributionTables);

        string summaryPath = Path.Combine(tables, "dataset_summary.csv");
        using (StreamWriter writer = new StreamWriter(summaryPath, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("object_count,accessed_object_count,event_count,log_file_count,total_object_bytes,total_object_tib,total_request_bytes,total_request_pib,first_access_bytes,start_time,end_time");
            writer.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "{0},{1},{2},{3},{4},{5:R},{6},{7:R},{8},{9},{10}",
                data.ObjectCount,
                data.AccessedObjectCount,
                data.EventCount,
                data.LogFileCount,
                data.TotalObjectBytes,
                data.TotalObjectBytes / Math.Pow(1024.0, 4),
                data.TotalRequestBytes,
                data.TotalRequestBytes / Math.Pow(1024.0, 5),
                data.FirstAccessBytes,
                data.StartTime.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                data.EndTime.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture)
            ));
        }

        string resultsPath = Path.Combine(tables, "lru_results.csv");
        using (StreamWriter writer = new StreamWriter(resultsPath, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("scenario,mode,channels,capacity_percent,capacity_bytes,capacity_tib,bandwidth_bytes_per_second,hit_events,io_hit_rate,hit_bytes,byte_hit_rate,miss_events,submitted_transfers,completed_transfers_at_log_end,queued_at_log_end,inflight_at_log_end,max_queued,final_cache_used_bytes,final_cache_object_count,evictions,evicted_bytes,refetch_transfers,refetch_bytes,validation_status");
            foreach (Result row in results)
            {
                long misses = data.EventCount - row.HitEvents;
                writer.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "{0},{1},{2},{3},{4},{5:R},{6},{7},{8:R},{9},{10:R},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},passed",
                    row.Scenario,
                    row.Mode,
                    row.Channels,
                    row.CapacityPercent,
                    row.CapacityBytes,
                    row.CapacityBytes / Math.Pow(1024.0, 4),
                    row.Mode == "instant" ? 0UL : BandwidthBytesPerSecond,
                    row.HitEvents,
                    (double)row.HitEvents / data.EventCount,
                    row.HitBytes,
                    (double)row.HitBytes / data.TotalRequestBytes,
                    misses,
                    row.SubmittedTransfers,
                    row.CompletedTransfersAtLogEnd,
                    row.QueuedAtLogEnd,
                    row.InflightAtLogEnd,
                    row.MaxQueued,
                    row.FinalCacheUsedBytes,
                    row.FinalCacheObjectCount,
                    row.Evictions,
                    row.EvictedBytes,
                    row.RefetchTransfers,
                    row.RefetchBytes
                ));
            }
        }

        string causesPath = Path.Combine(attributionTables, "miss_cause_summary.csv");
        using (StreamWriter writer = new StreamWriter(causesPath, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("scenario,mode,channels,capacity_percent,cause,miss_events,cause_events,event_share,miss_bytes,cause_bytes,byte_share");
            foreach (Result row in results)
            {
                long misses = data.EventCount - row.HitEvents;
                ulong missBytes = data.TotalRequestBytes - row.HitBytes;
                for (int i = 0; i < CauseCodes.Length; i++)
                    writer.WriteLine(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6},{7:R},{8},{9},{10:R}",
                        Csv(row.Scenario), row.Mode, row.Channels, row.CapacityPercent, CauseCodes[i],
                        misses, row.Attribution.CauseEvents[i],
                        misses == 0 ? 0.0 : (double)row.Attribution.CauseEvents[i] / misses,
                        missBytes, row.Attribution.CauseBytes[i],
                        missBytes == 0 ? 0.0 : (double)row.Attribution.CauseBytes[i] / missBytes
                    ));
            }
        }

        string operationalPath = Path.Combine(attributionTables, "miss_operational_state.csv");
        using (StreamWriter writer = new StreamWriter(operationalPath, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("scenario,channels,capacity_percent,operational_state,miss_events,state_events,event_share,miss_bytes,state_bytes,byte_share");
            foreach (Result row in results)
            {
                if (row.Mode != "finite") continue;
                long misses = data.EventCount - row.HitEvents;
                ulong missBytes = data.TotalRequestBytes - row.HitBytes;
                for (int i = 0; i < OperationalCodes.Length; i++)
                    writer.WriteLine(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6:R},{7},{8},{9:R}",
                        Csv(row.Scenario), row.Channels, row.CapacityPercent, OperationalCodes[i],
                        misses, row.Attribution.OperationalEvents[i],
                        (double)row.Attribution.OperationalEvents[i] / misses,
                        missBytes, row.Attribution.OperationalBytes[i],
                        (double)row.Attribution.OperationalBytes[i] / missBytes
                    ));
            }
        }

        string waitPath = Path.Combine(attributionTables, "miss_wait_buckets.csv");
        using (StreamWriter writer = new StreamWriter(waitPath, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("scenario,channels,capacity_percent,wait_bucket,miss_events,bucket_events,event_share,miss_bytes,bucket_bytes,byte_share");
            foreach (Result row in results)
            {
                if (row.Mode != "finite") continue;
                long misses = data.EventCount - row.HitEvents;
                ulong missBytes = data.TotalRequestBytes - row.HitBytes;
                for (int i = 0; i < WaitBucketCodes.Length; i++)
                    writer.WriteLine(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6:R},{7},{8},{9:R}",
                        Csv(row.Scenario), row.Channels, row.CapacityPercent, WaitBucketCodes[i],
                        misses, row.Attribution.WaitBucketEvents[i],
                        (double)row.Attribution.WaitBucketEvents[i] / misses,
                        missBytes, row.Attribution.WaitBucketBytes[i],
                        (double)row.Attribution.WaitBucketBytes[i] / missBytes
                    ));
            }
        }

        string diagnosticsPath = Path.Combine(attributionTables, "miss_diagnostics.csv");
        using (StreamWriter writer = new StreamWriter(diagnosticsPath, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("scenario,mode,channels,capacity_percent,miss_events,miss_bytes,evictions,evicted_bytes,submitted_transfers,refetch_transfers,refetch_bytes,refetch_transfer_share,mean_wait_seconds,max_wait_seconds,counterfactual_finite_only_hits,counterfactual_finite_only_hit_bytes,validation_status");
            foreach (Result row in results)
            {
                long misses = data.EventCount - row.HitEvents;
                ulong missBytes = data.TotalRequestBytes - row.HitBytes;
                writer.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11:R},{12:R},{13:R},{14},{15},passed",
                    Csv(row.Scenario), row.Mode, row.Channels, row.CapacityPercent,
                    misses, missBytes, row.Evictions, row.EvictedBytes,
                    row.SubmittedTransfers, row.RefetchTransfers, row.RefetchBytes,
                    row.SubmittedTransfers == 0 ? 0.0 : (double)row.RefetchTransfers / row.SubmittedTransfers,
                    misses == 0 ? 0.0 : row.Attribution.TotalWaitSeconds / misses,
                    row.Attribution.MaxWaitSeconds,
                    row.Attribution.CounterfactualFiniteOnlyHits,
                    row.Attribution.CounterfactualFiniteOnlyHitBytes
                ));
            }
        }

        string topPath = Path.Combine(attributionTables, "miss_top_objects.csv");
        using (StreamWriter writer = new StreamWriter(topPath, false, new UTF8Encoding(true)))
        {
            writer.WriteLine("scenario,mode,channels,capacity_percent,rank,path_index,path,size_bytes,total_accesses,miss_events,miss_event_share,miss_byte_proxy,cold_start_events,capacity_eviction_events,oversize_events,bandwidth_queue_events,bandwidth_transfer_events,bandwidth_state_divergence_events");
            foreach (Result row in results)
            {
                long misses = data.EventCount - row.HitEvents;
                List<int> ids = new List<int>();
                for (int id = 0; id < data.ObjectCount; id++)
                    if (row.Attribution.ObjectMissEvents[id] > 0) ids.Add(id);
                ids.Sort(delegate(int left, int right) {
                    int byMiss = row.Attribution.ObjectMissEvents[right].CompareTo(row.Attribution.ObjectMissEvents[left]);
                    return byMiss != 0 ? byMiss : left.CompareTo(right);
                });
                int take = Math.Min(20, ids.Count);
                for (int rank = 0; rank < take; rank++)
                {
                    int id = ids[rank];
                    long objectMisses = row.Attribution.ObjectMissEvents[id];
                    ulong objectMissBytes = checked((ulong)objectMisses * data.Sizes[id]);
                    writer.WriteLine(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10:R},{11},{12},{13},{14},{15},{16},{17}",
                        Csv(row.Scenario), row.Mode, row.Channels, row.CapacityPercent,
                        rank + 1, id, Csv(data.Paths[id]), data.Sizes[id], data.ExpectedAccessCounts[id],
                        objectMisses, (double)objectMisses / misses, objectMissBytes,
                        row.Attribution.ObjectCauseEvents[0][id],
                        row.Attribution.ObjectCauseEvents[1][id],
                        row.Attribution.ObjectCauseEvents[2][id],
                        row.Attribution.ObjectCauseEvents[3][id],
                        row.Attribution.ObjectCauseEvents[4][id],
                        row.Attribution.ObjectCauseEvents[5][id]
                    ));
                }
            }
        }
        File.Copy(summaryPath, Path.Combine(attributionTables, "dataset_summary.csv"), true);
        File.Copy(resultsPath, Path.Combine(attributionTables, "lru_results.csv"), true);
        Console.WriteLine("summary: " + summaryPath);
        Console.WriteLine("results: " + resultsPath);
        Console.WriteLine("miss causes: " + causesPath);
        Console.WriteLine("miss operations: " + operationalPath);
        Console.WriteLine("miss waits: " + waitPath);
        Console.WriteLine("miss diagnostics: " + diagnosticsPath);
        Console.WriteLine("miss top objects: " + topPath);
    }

    public static void Run(string root)
    {
        Console.OutputEncoding = Encoding.UTF8;
        DateTime allStarted = DateTime.UtcNow;
        Dataset data = ReadCatalog(root);
        Console.WriteLine(
            "objects={0:N0}, events={1:N0}, total={2:F2} TiB, weighted requests={3:F2} PiB",
            data.ObjectCount,
            data.EventCount,
            data.TotalObjectBytes / Math.Pow(1024.0, 4),
            data.TotalRequestBytes / Math.Pow(1024.0, 5)
        );
        ReadEvents(root, data);
        Console.WriteLine("trace: {0} -> {1}", data.StartTime, data.EndTime);

        List<Result> results = new List<Result>();
        byte[][] instantCauses = new byte[CapacityPercents.Length][];
        for (int i = 0; i < CapacityPercents.Length; i++)
        {
            int percent = CapacityPercents[i];
            ulong capacity = data.TotalObjectBytes * (ulong)percent / 100UL;
            byte[] causes;
            Result result = SimulateInstant(data, percent, capacity, out causes);
            instantCauses[i] = causes;
            ValidateResult(data, result);
            results.Add(result);
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        for (int c = 0; c < ChannelCounts.Length; c++)
        {
            int channels = ChannelCounts[c];
            for (int i = 0; i < CapacityPercents.Length; i++)
            {
                int percent = CapacityPercents[i];
                ulong capacity = data.TotalObjectBytes * (ulong)percent / 100UL;
                Result result = SimulateLimited(data, channels, percent, capacity, instantCauses[i]);
                ValidateResult(data, result);
                results.Add(result);
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        WriteOutputs(root, data, results);
        Console.WriteLine("all done in {0:F1}s", (DateTime.UtcNow - allStarted).TotalSeconds);
    }
}

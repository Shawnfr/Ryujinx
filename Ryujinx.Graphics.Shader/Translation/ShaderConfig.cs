using System;
using System.Collections.Generic;

namespace Ryujinx.Graphics.Shader.Translation
{
    class ShaderConfig
    {
        public ShaderStage Stage { get; }

        public bool GpPassthrough { get; }

        public OutputTopology OutputTopology { get; }

        public int MaxOutputVertices { get; }

        public int LocalMemorySize { get; }

        public ImapPixelType[] ImapTypes { get; }

        public OmapTarget[] OmapTargets    { get; }
        public bool         OmapSampleMask { get; }
        public bool         OmapDepth      { get; }

        public IGpuAccessor GpuAccessor { get; }

        public TranslationFlags Flags { get; }

        public TranslationCounts Counts { get; }

        public int Size { get; private set; }

        public byte ClipDistancesWritten { get; private set; }

        public FeatureFlags UsedFeatures { get; private set; }

        public HashSet<int> TextureHandlesForCache { get; }

        private readonly Dictionary<int, int> _sbSlots;
        private readonly Dictionary<int, int> _sbSlotsReverse;

        public ShaderConfig(IGpuAccessor gpuAccessor, TranslationFlags flags, TranslationCounts counts)
        {
            Stage                  = ShaderStage.Compute;
            GpPassthrough          = false;
            OutputTopology         = OutputTopology.PointList;
            MaxOutputVertices      = 0;
            LocalMemorySize        = 0;
            ImapTypes              = null;
            OmapTargets            = null;
            OmapSampleMask         = false;
            OmapDepth              = false;
            GpuAccessor            = gpuAccessor;
            Flags                  = flags;
            Size                   = 0;
            UsedFeatures           = FeatureFlags.None;
            Counts                 = counts;
            TextureHandlesForCache = new HashSet<int>();
            _sbSlots               = new Dictionary<int, int>();
            _sbSlotsReverse        = new Dictionary<int, int>();
        }

        public ShaderConfig(ShaderHeader header, IGpuAccessor gpuAccessor, TranslationFlags flags, TranslationCounts counts) : this(gpuAccessor, flags, counts)
        {
            Stage             = header.Stage;
            GpPassthrough     = header.Stage == ShaderStage.Geometry && header.GpPassthrough;
            OutputTopology    = header.OutputTopology;
            MaxOutputVertices = header.MaxOutputVertexCount;
            LocalMemorySize   = header.ShaderLocalMemoryLowSize + header.ShaderLocalMemoryHighSize;
            ImapTypes         = header.ImapTypes;
            OmapTargets       = header.OmapTargets;
            OmapSampleMask    = header.OmapSampleMask;
            OmapDepth         = header.OmapDepth;
        }

        public int GetDepthRegister()
        {
            int count = 0;

            for (int index = 0; index < OmapTargets.Length; index++)
            {
                for (int component = 0; component < 4; component++)
                {
                    if (OmapTargets[index].ComponentEnabled(component))
                    {
                        count++;
                    }
                }
            }

            // The depth register is always two registers after the last color output.
            return count + 1;
        }

        public TextureFormat GetTextureFormat(int handle)
        {
            // When the formatted load extension is supported, we don't need to
            // specify a format, we can just declare it without a format and the GPU will handle it.
            if (GpuAccessor.QuerySupportsImageLoadFormatted())
            {
                return TextureFormat.Unknown;
            }

            var format = GpuAccessor.QueryTextureFormat(handle);

            if (format == TextureFormat.Unknown)
            {
                GpuAccessor.Log($"Unknown format for texture {handle}.");

                format = TextureFormat.R8G8B8A8Unorm;
            }

            return format;
        }

        public void SizeAdd(int size)
        {
            Size += size;
        }

        public void SetClipDistanceWritten(int index)
        {
            ClipDistancesWritten |= (byte)(1 << index);
        }

        public void SetUsedFeature(FeatureFlags flags)
        {
            UsedFeatures |= flags;
        }

        public int GetSbSlot(byte sbCbSlot, ushort sbCbOffset)
        {
            int key = PackSbCbInfo(sbCbSlot, sbCbOffset);

            if (!_sbSlots.TryGetValue(key, out int slot))
            {
                slot = _sbSlots.Count;
                _sbSlots.Add(key, slot);
                _sbSlotsReverse.Add(slot, key);
            }

            return slot;
        }

        public (int, int) GetSbCbInfo(int slot)
        {
            if (_sbSlotsReverse.TryGetValue(slot, out int key))
            {
                return UnpackSbCbInfo(key);
            }

            throw new ArgumentException($"Invalid slot {slot}.", nameof(slot));
        }

        public BufferDescriptor GetSbDescriptor(int binding, int slot)
        {
            if (_sbSlotsReverse.TryGetValue(slot, out int key))
            {
                (int sbCbSlot, int sbCbOffset) = UnpackSbCbInfo(key);

                return new BufferDescriptor(binding, slot, sbCbSlot, sbCbOffset);
            }

            throw new ArgumentException($"Invalid slot {slot}.", nameof(slot));
        }

        private static int PackSbCbInfo(int sbCbSlot, int sbCbOffset)
        {
            return sbCbOffset | ((int)sbCbSlot << 16);
        }

        private static (int, int) UnpackSbCbInfo(int key)
        {
            return ((byte)(key >> 16), (ushort)key);
        }
    }
}
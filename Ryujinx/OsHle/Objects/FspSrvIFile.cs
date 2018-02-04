using ChocolArm64.Memory;
using System;

using System.IO;

namespace Ryujinx.OsHle.Objects
{
    class FspSrvIFile : IDisposable
    {
        public Stream BaseStream { get; private set; }

        public FspSrvIFile(Stream BaseStream)
        {
            this.BaseStream = BaseStream;
        }

        public static long Read(ServiceCtx Context)
        {
            FspSrvIFile File = Context.GetObject<FspSrvIFile>();

            long Position = Context.Request.ReceiveBuff[0].Position;

            long Zero   = Context.RequestData.ReadInt64();
            long Offset = Context.RequestData.ReadInt64();
            long Size   = Context.RequestData.ReadInt64();

            byte[] Data = new byte[Size];

            int ReadSize = File.BaseStream.Read(Data, 0, (int)Size);

            AMemoryHelper.WriteBytes(Context.Memory, Position, Data);

            //TODO: Use ReadSize, we need to return the size that was REALLY read from the file.
            //This is a workaround because we are doing something wrong and the game expects to read
            //data from a file that doesn't yet exists -- and breaks if it can't read anything.
            Context.ResponseData.Write((long)Size);

            return 0;
        }

        public static long Write(ServiceCtx Context)
        {
            FspSrvIFile File = Context.GetObject<FspSrvIFile>();

            long Position = Context.Request.SendBuff[0].Position;

            long Zero   = Context.RequestData.ReadInt64();
            long Offset = Context.RequestData.ReadInt64();
            long Size   = Context.RequestData.ReadInt64();

            byte[] Data = AMemoryHelper.ReadBytes(Context.Memory, Position, (int)Size);

            File.BaseStream.Seek(Offset, SeekOrigin.Begin);
            File.BaseStream.Write(Data, 0, (int)Size);

            return 0;
        }

        public void Dispose()
        {
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing && BaseStream != null)
            {
                BaseStream.Dispose();
            }
        }
    }
}
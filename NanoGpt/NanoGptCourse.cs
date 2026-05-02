namespace ukazure.llm.NanoGpt;

internal sealed class NanoGptCourse : ILesson
{
    public string Title => "nanoGPT in C# with TorchSharp";

    public void Run()
    {
        var run = TorchSharpNanoGptRunner.Run();
        NanoGptPresenter.Run(run);
    }
}

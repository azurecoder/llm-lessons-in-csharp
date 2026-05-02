using ukazure.llm;
using ukazure.llm.Lesson1;
using ukazure.llm.Lesson2;
using ukazure.llm.Lesson3;
using ukazure.llm.Lesson4;
using ukazure.llm.Lesson5;
using ukazure.llm.Lesson6;
using ukazure.llm.Lesson7;
using ukazure.llm.NanoGpt;

var commands = new Dictionary<string, ILesson>(StringComparer.OrdinalIgnoreCase)
{
    ["lesson1"] = new Lesson1Course(),
    ["lesson2"] = new Lesson2Course(),
    ["lesson3"] = new Lesson3Course(),
    ["lesson4"] = new Lesson4Course(),
    ["lesson5"] = new Lesson5Course(),
    ["lesson6"] = new Lesson6Course(),
    ["lesson7"] = new Lesson7Course(),
    ["nanogpt"] = new NanoGptCourse()
};

if (args.Length != 1 || !commands.TryGetValue(args[0], out var command))
{
    Console.WriteLine("Usage: dotnet run lesson1|lesson2|lesson3|lesson4|lesson5|lesson6|lesson7|nanogpt");
    Console.WriteLine();
    Console.WriteLine("Available commands:");

    foreach (var entry in commands.OrderBy(entry => entry.Key))
    {
        Console.WriteLine($"  {entry.Key,-8} {entry.Value.Title}");
    }

    return;
    
}

command.Run();

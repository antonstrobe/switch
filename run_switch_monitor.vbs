Set shell = CreateObject("WScript.Shell")
root = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
pythonw = shell.ExpandEnvironmentStrings("%LocalAppData%") & "\Programs\Python\Python312\pythonw.exe"

If CreateObject("Scripting.FileSystemObject").FileExists(pythonw) Then
  shell.Run """" & pythonw & """ """ & root & "\app.pyw""", 0, False
Else
  shell.Run "pyw -3 """ & root & "\app.pyw""", 0, False
End If

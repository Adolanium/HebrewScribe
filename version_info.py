# pyright: reportUndefinedVariable=false

VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=(2, 0, 0, 0),
        prodvers=(2, 0, 0, 0),
        mask=0x3F,
        flags=0x0,
        OS=0x40004,          # VOS_NT_WINDOWS32
        fileType=0x1,        # VFT_APP
        subtype=0x0,
    ),
    kids=[
        StringFileInfo([
            StringTable(
                '040904B0',  # Lang: US English, Charset: Unicode
                [
                    StringStruct('CompanyName', 'Yevgeniy Glider'),
                    StringStruct('FileDescription', 'HebrewScribe'),
                    StringStruct('FileVersion', '2.0.0.0'),
                    StringStruct('InternalName', 'HebrewScribe'),
                    StringStruct('OriginalFilename', 'HebrewScribe.exe'),
                    StringStruct('ProductName', 'HebrewScribe'),
                    StringStruct('ProductVersion', '2.0.0.0'),
                ],
            ),
        ]),
        VarFileInfo([VarStruct('Translation', [0x0409, 1200])]),
    ],
)

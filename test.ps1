
function play {
    param (
        [String] $black_cmd, [String] $white_cmd
    )
    if (Test-Path -Path ".\input.txt" -PathType Leaf) {
        Remove-Item .\input.txt
    }

    if (Test-Path -Path ".\output.txt" -PathType Leaf) {
        Remove-Item .\output.txt
    }

    Copy-Item -Path ".\input-init.txt" -Destination ".\input.txt"

    $moves = 0
    $black_time = 0
    $white_time = 0
    while (1) {

        if (Test-Path -Path ".\output.txt" -PathType Leaf) {
            Remove-Item .\output.txt
        }

        $black_time += Measure-Command { Invoke-Expression $black_cmd | Out-Host }
        $moves += 1
        python host.py -m $moves
        if (!$?) {
            break
        }
        if (Test-Path -Path ".\output.txt" -PathType Leaf) {
            Remove-Item .\output.txt
        }
        $white_time += Measure-Command { Invoke-Expression $white_cmd | Out-Host }
        $moves += 1
        python host.py -m $moves
        if (!$?) {
            break
        }
    }

    Write-Output "Black: $([Math]::Round($black_time.TotalSeconds/ [int][Math]::Ceiling($moves / 2),2))s per move, White: $([Math]::Round($white_time.TotalSeconds/ [int][Math]::Floor($moves / 2),2))s per move"
}

$play_time = 10
$cmd = "python my_player3.py"
$opponent_cmd = "python random_player.py"
$black_win_time = 0
$white_win_time = 0
$black_tie = 0
$white_tie = 0

for ($i = 1; $i -le $play_time; $i += 2) {
    Write-Output "=====Round $i====="
    Write-Output "Your Colour: Black"
    play "$cmd" "$opponent_cmd"
    if ($LastExitCode -eq 1) {
        Write-Output 'You win!'
        $black_win_time += 1
    }
    elseif ($LastExitCode -eq 0) {
        Write-Output Tie.
        $black_tie += 1
    }
    else {
        Write-Output 'You lose.'
    }

    # Student takes White
    Write-Output "=====Round $($i+1)====="
    Write-Output "Your Colour: White"

    play "$opponent_cmd" "$cmd"
    if ($LastExitCode -eq 2) {
        Write-Output 'You win!'
        $white_win_time += 1
    }
    elseif ($LastExitCode -eq 0) {
        Write-Output Tie.
        $white_tie += 1
    }
    else {
        Write-Output 'You lose.'
    }


}

Write-Output =====Summary=====
Write-Output "You play as Black Player | Win: $black_win_time | Lose: $(($play_time/2-$black_win_time-$black_tie)) | Tie: $black_tie"
Write-Output "You play as White Player | Win: $white_win_time | Lose: $(($play_time/2-$white_win_time-$black_tie)) | Tie: $white_tie"